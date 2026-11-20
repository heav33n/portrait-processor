//! The canvas is fixed. Only the foreground layer is transformed or trimmed.
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::VecDeque;

const WIDTH: usize = 948;
const HEIGHT: usize = 1048;
const CENTER: f64 = (WIDTH as f64 - 1.0) / 2.0;

#[derive(Clone, Copy)]
struct Options {
    nose: (f64, f64),
    angle: f64,
    headroom: u32,
    side: u32,
    bottom: u32,
    symmetry: bool,
    horizontal_offset: i32,
}

struct Rendered {
    rgb: Vec<u8>,
    before_rgb: Vec<u8>,
    alpha: Vec<u8>,
    before_alpha: Vec<u8>,
    transform: [f64; 6],
    inverse: [f64; 6],
    scale: f64,
    removed_fraction: f64,
}

fn placement(alpha: &[u8], w: usize, h: usize, o: Options) -> Result<([f64; 6], f64), String> {
    if w == 0 || h == 0 || alpha.len() != w * h {
        return Err("Image and mask dimensions must match and be nonzero.".into());
    }
    if !o.nose.0.is_finite() || !o.nose.1.is_finite() || !o.angle.is_finite() {
        return Err("Nose coordinates and rotation must be finite.".into());
    }
    if o.nose.0 < 0.0 || o.nose.0 > (w - 1) as f64 || o.nose.1 < 0.0 || o.nose.1 > (h - 1) as f64 {
        return Err("Place the nose inside the source image.".into());
    }
    if u64::from(o.headroom) + u64::from(o.bottom) >= HEIGHT as u64
        || u64::from(o.side) * 2 >= WIDTH as u64
    {
        return Err("Margins leave no room for the portrait.".into());
    }
    let (sin, cos) = o.angle.to_radians().sin_cos();
    let mut bounds = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for (i, &a) in alpha.iter().enumerate() {
        if a == 0 {
            continue;
        }
        let x = (i % w) as f64 - o.nose.0;
        let y = (i / w) as f64 - o.nose.1;
        let rx = cos * x - sin * y;
        let ry = sin * x + cos * y;
        bounds[0] = bounds[0].min(rx);
        bounds[1] = bounds[1].max(rx);
        bounds[2] = bounds[2].min(ry);
        bounds[3] = bounds[3].max(ry);
    }
    if !bounds[0].is_finite() {
        return Err("The foreground mask is empty. Restore part of the portrait first.".into());
    }
    // Bounds include complete source pixels, not just their centers.
    let half = (sin.abs() + cos.abs()) * 0.5;
    let center = CENTER + o.horizontal_offset as f64;
    let left_room = center + 0.5 - o.side as f64;
    let right_room = WIDTH as f64 - o.side as f64 - 0.5 - center;
    if left_room <= 0.0 || right_room <= 0.0 {
        return Err("The left/right adjustment must stay inside the side guides.".into());
    }
    let top = bounds[2] - half;
    let tall = bounds[3] + half - top;
    let scale = 1.0_f64
        .min(left_room / (half - bounds[0]).max(f64::EPSILON))
        .min(right_room / (bounds[1] + half).max(f64::EPSILON))
        .min((HEIGHT as f64 - o.headroom as f64 - o.bottom as f64) / tall);
    // Build upward from the bottom edge. Headroom is an upper boundary, not a top anchor.
    let vertical = HEIGHT as f64 - o.bottom as f64 - 0.5 - scale * (bounds[3] + half);
    let a = scale * cos;
    let b = -scale * sin;
    let d = scale * sin;
    let e = scale * cos;
    Ok((
        [
            a,
            b,
            center - a * o.nose.0 - b * o.nose.1,
            d,
            e,
            vertical - d * o.nose.0 - e * o.nose.1,
        ],
        scale,
    ))
}

fn invert(t: [f64; 6]) -> [f64; 6] {
    let det = t[0] * t[4] - t[1] * t[3];
    [
        t[4] / det,
        -t[1] / det,
        (t[1] * t[5] - t[4] * t[2]) / det,
        -t[3] / det,
        t[0] / det,
        (t[3] * t[2] - t[0] * t[5]) / det,
    ]
}

// Interpolate premultiplied colors so hidden background colors cannot bleed into edges.
fn sample(rgb: &[u8], alpha: &[u8], w: usize, h: usize, x: f64, y: f64) -> [f64; 4] {
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;
    let fx = x - ix as f64;
    let fy = y - iy as f64;
    let mut out = [0.0; 4];
    for (dx, dy, weight) in [
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ] {
        let sx = ix + dx;
        let sy = iy + dy;
        if sx < 0 || sy < 0 || sx >= w as isize || sy >= h as isize {
            continue;
        }
        let i = sy as usize * w + sx as usize;
        let a = alpha[i] as f64 / 255.0 * weight;
        out[3] += a;
        for c in 0..3 {
            out[c] += rgb[i * 3 + c] as f64 * a;
        }
    }
    out
}

fn paired_alpha(alpha: &[u8], w: usize, offset: i32) -> Vec<u8> {
    alpha
        .par_chunks(w)
        .flat_map_iter(|row| {
            row.iter()
                .enumerate()
                .map(|(x, &a)| {
                    let reflected = w as i64 - 1 + 2 * offset as i64 - x as i64;
                    if reflected < 0 || reflected >= w as i64 {
                        0
                    } else {
                        a.min(row[reflected as usize])
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn bottom_anchor(alpha: &[u8], w: usize) -> Option<usize> {
    let last = alpha.iter().rposition(|&a| a > 0)? / w;
    // A rotated or feathered cut edge ends in faint pixels. Anchor to a broadly
    // covered row just above that fringe, using opacity rather than RGB brightness.
    let first = last.saturating_sub(159);
    let band = &alpha[first * w..(last + 1) * w];
    let maximum = *band.iter().max()? as usize;
    let threshold = (maximum * 98).div_ceil(100).clamp(1, 250) as u8;
    let coverage: Vec<usize> = band
        .chunks(w)
        .map(|row| row.iter().filter(|&&a| a >= threshold).count())
        .collect();
    let widest = *coverage.iter().max()?;
    // A 95% row still leaves a visible white wedge below a tilted photo edge.
    // Crop at the last full-width solid row, without judging clothing brightness.
    let required = widest;
    coverage
        .iter()
        .rposition(|&count| count >= required)
        .map(|row| first + row)
}

fn render_core(
    rgb: &[u8],
    alpha: &[u8],
    w: usize,
    h: usize,
    o: Options,
) -> Result<Rendered, String> {
    if rgb.len() != w * h * 3 {
        return Err("Expected an RGB image with three channels.".into());
    }
    let (mut transform, scale) = placement(alpha, w, h, o)?;
    let mut inverse = invert(transform);
    // Area supersampling when shrinking, capped to keep very large inputs responsive.
    let steps = (1.0 / scale).ceil().clamp(1.0, 8.0) as usize;
    let mut pixels = vec![[0.0; 4]; WIDTH * HEIGHT];
    pixels
        .par_chunks_mut(WIDTH)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, dst) in row.iter_mut().enumerate() {
                for sy in 0..steps {
                    for sx in 0..steps {
                        let px = x as f64 + (sx as f64 + 0.5) / steps as f64 - 0.5;
                        let py = y as f64 + (sy as f64 + 0.5) / steps as f64 - 0.5;
                        let value = sample(
                            rgb,
                            alpha,
                            w,
                            h,
                            inverse[0] * px + inverse[1] * py + inverse[2],
                            inverse[3] * px + inverse[4] * py + inverse[5],
                        );
                        for c in 0..4 {
                            dst[c] += value[c] / (steps * steps) as f64;
                        }
                    }
                }
            }
        });
    let mut before_alpha: Vec<u8> = pixels
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (x, y) = (i % WIDTH, i / WIDTH);
            if y < o.headroom as usize
                || y >= HEIGHT - o.bottom as usize
                || x < o.side as usize
                || x >= WIDTH - o.side as usize
            {
                0
            } else {
                (p[3] * 255.0).round().clamp(0.0, 255.0) as u8
            }
        })
        .collect();
    let mut final_alpha = if o.symmetry {
        paired_alpha(&before_alpha, WIDTH, o.horizontal_offset)
    } else {
        before_alpha.clone()
    };
    // Move the solid lower edge to the boundary and crop only its faint fringe.
    // Whole-pixel translation preserves hair edges, texture, and source colors.
    if let Some(last_row) = bottom_anchor(&final_alpha, WIDTH) {
        let shift = HEIGHT - o.bottom as usize - 1 - last_row;
        if shift > 0 {
            let offset = shift * WIDTH;
            let retained = (HEIGHT - shift) * WIDTH;
            pixels.copy_within(..retained, offset);
            pixels[..offset].fill([0.0; 4]);
            before_alpha.copy_within(..retained, offset);
            before_alpha[..offset].fill(0);
            before_alpha[(HEIGHT - o.bottom as usize) * WIDTH..].fill(0);
            final_alpha.copy_within(..retained, offset);
            final_alpha[..offset].fill(0);
            final_alpha[(HEIGHT - o.bottom as usize) * WIDTH..].fill(0);
            transform[5] += shift as f64;
            inverse = invert(transform);
        }
    }
    let mut before_rgb = vec![255; WIDTH * HEIGHT * 3];
    let mut result = before_rgb.clone();
    result
        .par_chunks_mut(3)
        .zip(before_rgb.par_chunks_mut(3))
        .enumerate()
        .for_each(|(i, (after, before))| {
            let p = pixels[i];
            if p[3] > 0.0 {
                for c in 0..3 {
                    let foreground = (p[c] / p[3]).clamp(0.0, 255.0);
                    before[c] = (foreground * (before_alpha[i] as f64 / 255.0) + 255.0
                        - before_alpha[i] as f64)
                        .round() as u8;
                    after[c] = (foreground * (final_alpha[i] as f64 / 255.0) + 255.0
                        - final_alpha[i] as f64)
                        .round() as u8;
                }
            }
        });
    let before_total: u64 = before_alpha.iter().map(|&a| a as u64).sum();
    let after_total: u64 = final_alpha.iter().map(|&a| a as u64).sum();
    Ok(Rendered {
        rgb: result,
        before_rgb,
        alpha: final_alpha,
        before_alpha,
        transform,
        inverse,
        scale,
        removed_fraction: if before_total == 0 {
            0.0
        } else {
            1.0 - after_total as f64 / before_total as f64
        },
    })
}

#[pyfunction]
#[pyo3(signature = (rgb, alpha, nose, angle_degrees=0.0, headroom=180, side_margin=64, bottom_margin=0, symmetry=true, horizontal_offset=0))]
#[allow(clippy::too_many_arguments)]
fn render<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, u8>,
    alpha: PyReadonlyArray2<'py, u8>,
    nose: (f64, f64),
    angle_degrees: f64,
    headroom: u32,
    side_margin: u32,
    bottom_margin: u32,
    symmetry: bool,
    horizontal_offset: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = rgb.shape();
    let (h, w) = (shape[0], shape[1]);
    if shape[2] != 3 || alpha.shape() != [h, w] {
        return Err(PyValueError::new_err(
            "RGB image and alpha mask dimensions do not match.",
        ));
    }
    let rgb = rgb.as_array().iter().copied().collect::<Vec<_>>();
    let alpha = alpha.as_array().iter().copied().collect::<Vec<_>>();
    let o = Options {
        nose,
        angle: angle_degrees,
        headroom,
        side: side_margin,
        bottom: bottom_margin,
        symmetry,
        horizontal_offset,
    };
    let r = py
        .detach(|| render_core(&rgb, &alpha, w, h, o))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item(
        "rgb",
        Array3::from_shape_vec((HEIGHT, WIDTH, 3), r.rgb)
            .unwrap()
            .into_pyarray(py),
    )?;
    out.set_item(
        "before_rgb",
        Array3::from_shape_vec((HEIGHT, WIDTH, 3), r.before_rgb)
            .unwrap()
            .into_pyarray(py),
    )?;
    out.set_item(
        "alpha",
        Array2::from_shape_vec((HEIGHT, WIDTH), r.alpha)
            .unwrap()
            .into_pyarray(py),
    )?;
    out.set_item(
        "before_alpha",
        Array2::from_shape_vec((HEIGHT, WIDTH), r.before_alpha)
            .unwrap()
            .into_pyarray(py),
    )?;
    out.set_item("transform", r.transform.to_vec())?;
    out.set_item("inverse", r.inverse.to_vec())?;
    out.set_item("scale", r.scale)?;
    out.set_item("removed_fraction", r.removed_fraction)?;
    Ok(out)
}

fn clean_core(
    alpha: &[u8],
    w: usize,
    h: usize,
    min_area: usize,
    anchor: Option<(f64, f64)>,
) -> Vec<u8> {
    let mut out = alpha.to_vec();
    let mut seen = vec![false; alpha.len()];
    let mut components = Vec::new();
    let mut closest = (f64::INFINITY, usize::MAX);
    for start in 0..alpha.len() {
        if seen[start] || alpha[start] == 0 {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        while let Some(i) = queue.pop_front() {
            component.push(i);
            let (x, y) = (i % w, i / w);
            if let Some((ax, ay)) = anchor {
                let d = (x as f64 - ax).powi(2) + (y as f64 - ay).powi(2);
                if d < closest.0 {
                    closest = (d, components.len());
                }
            }
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let n = ny as usize * w + nx as usize;
                    if !seen[n] && alpha[n] > 0 {
                        seen[n] = true;
                        queue.push_back(n);
                    }
                }
            }
        }
        components.push(component);
    }
    for (id, component) in components.into_iter().enumerate() {
        if id != closest.1 && component.len() <= min_area {
            for i in component {
                out[i] = 0;
            }
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (alpha, min_area=24, anchor=None))]
fn clean_specks<'py>(
    py: Python<'py>,
    alpha: PyReadonlyArray2<'py, u8>,
    min_area: usize,
    anchor: Option<(f64, f64)>,
) -> PyResult<Bound<'py, numpy::PyArray2<u8>>> {
    let (h, w) = (alpha.shape()[0], alpha.shape()[1]);
    let data = alpha.as_array().iter().copied().collect::<Vec<_>>();
    let cleaned = py.detach(|| clean_core(&data, w, h, min_area, anchor));
    Ok(Array2::from_shape_vec((h, w), cleaned)
        .unwrap()
        .into_pyarray(py))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render, m)?)?;
    m.add_function(wrap_pyfunction!(clean_specks, m)?)?;
    m.add("WIDTH", WIDTH)?;
    m.add("HEIGHT", HEIGHT)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> Options {
        Options {
            nose: (4.0, 4.0),
            angle: 0.0,
            headroom: 60,
            side: 24,
            bottom: 24,
            symmetry: true,
            horizontal_offset: 0,
        }
    }
    #[test]
    fn symmetry_only_removes_and_matches_every_pair() {
        let source = vec![0, 10, 200, 255, 20, 0, 255, 150, 0, 0];
        let trimmed = paired_alpha(&source, 5, 0);
        for (row, original) in trimmed.chunks(5).zip(source.chunks(5)) {
            assert!(row.iter().eq(row.iter().rev()));
            assert!(row.iter().zip(original).all(|(a, b)| a <= b));
        }
    }
    #[test]
    fn centers_nose_and_never_enlarges() {
        let (t, scale) = placement(&[255; 100], 10, 10, options()).unwrap();
        assert_eq!(scale, 1.0);
        assert!((t[0] * 4.0 + t[1] * 4.0 + t[2] - CENTER).abs() < 1e-9);
        let inv = invert(t);
        let x = t[0] * 2.0 + t[1] * 3.0 + t[2];
        let y = t[3] * 2.0 + t[4] * 3.0 + t[5];
        assert!((inv[0] * x + inv[1] * y + inv[2] - 2.0).abs() < 1e-9);
    }
    #[test]
    fn small_subject_builds_up_from_bottom() {
        let o = Options {
            bottom: 0,
            ..options()
        };
        let (t, scale) = placement(&[255; 100], 10, 10, o).unwrap();
        assert_eq!(scale, 1.0);
        assert!((t[3] * 4.0 + t[4] * 9.0 + t[5] - 1047.0).abs() < 1e-9);
    }
    #[test]
    fn bottom_anchor_ignores_faint_feathered_rows() {
        let mask = vec![255, 255, 255, 255, 180, 180, 180, 180, 10, 10, 0, 0];
        assert_eq!(bottom_anchor(&mask, 4), Some(0));
        assert_eq!(bottom_anchor(&[0; 12], 4), None);
        assert_eq!(bottom_anchor(&[80; 12], 4), Some(2));
    }
    #[test]
    fn bottom_anchor_crops_a_slanted_cut_edge_completely() {
        let mut mask = vec![0; 100 * 200];
        for y in 0..200 {
            for x in 0..100 {
                if y < 120 + x / 2 {
                    mask[y * 100 + x] = 255;
                }
            }
        }
        assert_eq!(bottom_anchor(&mask, 100), Some(119));
    }
    #[test]
    fn shrinks_large_portrait_and_rejects_empty_masks() {
        let (_, scale) = placement(
            &vec![255; 2000 * 2000],
            2000,
            2000,
            Options {
                nose: (999.5, 999.5),
                ..options()
            },
        )
        .unwrap();
        assert!(scale < 0.5);
        assert!(placement(&[0; 100], 10, 10, options()).is_err());
        assert!(placement(
            &[255; 100],
            10,
            10,
            Options {
                headroom: 1048,
                ..options()
            }
        )
        .is_err());
    }
    #[test]
    fn cleanup_preserves_face_component() {
        let mask = [255, 0, 0, 0, 255, 0, 0, 0, 0, 0];
        assert_eq!(
            clean_core(&mask, 5, 2, 2, Some((0.0, 0.0))),
            vec![255, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }
}
