#include <windows.h>
#include <winbase.h>
#include "fortran.h"


BOOL WINAPI
DllMain (HANDLE hDll, DWORD dwReason, LPVOID lpReserved)
{
    switch (dwReason)
    {
        case DLL_PROCESS_ATTACH:
			cublas_init_();// Code to run when the DLL is loaded
            break;

        case DLL_PROCESS_DETACH:
            ;// Code to run when the DLL is freed
            break;

        case DLL_THREAD_ATTACH:
            ;// Code to run when a thread is created during the DLL's lifetime
            break;

        case DLL_THREAD_DETACH:
            ;// Code to run when a thread ends normally.
            break;
    }
    return TRUE;
}

