#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <wchar.h>
#include <stdlib.h>

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR arguments, int show) {
    (void)instance; (void)previous; (void)show;
    wchar_t *directory = calloc(32768, sizeof(wchar_t));
    wchar_t *python = calloc(32768, sizeof(wchar_t));
    wchar_t *command = calloc(32768, sizeof(wchar_t));
    if (!directory || !python || !command) return 1;
    DWORD length = GetModuleFileNameW(NULL, directory, 32768);
    if (!length || length >= 32768) return 1;
    wchar_t *separator = wcsrchr(directory, L'\\');
    if (!separator) return 1;
    *separator = L'\0';
    if (swprintf(python, 32768, L"%ls\\runtime\\pythonw.exe", directory) < 0 ||
        swprintf(command, 32768, L"\"%ls\" \"%ls\\runtime\\start_app.py\" %ls",
                 python, directory, arguments) < 0) return 1;
    STARTUPINFOW startup = {0}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION process = {0};
    if (!CreateProcessW(python, command, NULL, NULL, FALSE, 0, NULL, directory, &startup, &process)) {
        MessageBoxW(NULL, L"Portrait Processor could not start. Extract the entire ZIP to a folder before opening the app.",
                    L"Portrait Processor", MB_OK | MB_ICONERROR);
        return 1;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hProcess);
    free(command); free(python); free(directory);
    return (int)exit_code;
}
