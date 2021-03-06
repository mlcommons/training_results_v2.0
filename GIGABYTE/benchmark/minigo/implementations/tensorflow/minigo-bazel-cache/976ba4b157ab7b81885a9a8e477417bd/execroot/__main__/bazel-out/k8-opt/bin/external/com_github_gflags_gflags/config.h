/* Generated from config.h.in during build configuration using CMake. */

// Note: This header file is only used internally. It is not part of public interface!

#ifndef GFLAGS_CONFIG_H_
#define GFLAGS_CONFIG_H_


// ---------------------------------------------------------------------------
// System checks

// Define if you build this library for a MS Windows OS.
//cmakedefine OS_WINDOWS

// Define if you have the <stdint.h> header file.
//cmakedefine HAVE_STDINT_H

// Define if you have the <sys/types.h> header file.
//cmakedefine HAVE_SYS_TYPES_H

// Define if you have the <inttypes.h> header file.
//cmakedefine HAVE_INTTYPES_H

// Define if you have the <sys/stat.h> header file.
//cmakedefine HAVE_SYS_STAT_H

// Define if you have the <unistd.h> header file.
//cmakedefine HAVE_UNISTD_H

// Define if you have the <fnmatch.h> header file.
//cmakedefine HAVE_FNMATCH_H

// Define if you have the <shlwapi.h> header file (Windows 2000/XP).
//cmakedefine HAVE_SHLWAPI_H

// Define if you have the strtoll function.
//cmakedefine HAVE_STRTOLL

// Define if you have the strtoq function.
//cmakedefine HAVE_STRTOQ

// Define if you have the <pthread.h> header file.
//cmakedefine HAVE_PTHREAD

// Define if your pthread library defines the type pthread_rwlock_t
//cmakedefine HAVE_RWLOCK

// gcc requires this to get PRId64, etc.
#if defined(HAVE_INTTYPES_H) && !defined(__STDC_FORMAT_MACROS)
#  define __STDC_FORMAT_MACROS 1
#endif

// ---------------------------------------------------------------------------
// Package information

// Name of package.
#define PACKAGE @PROJECT_NAME@

// Define to the full name of this package.
#define PACKAGE_NAME @PACKAGE_NAME@

// Define to the full name and version of this package.
#define PACKAGE_STRING @PACKAGE_STRING@

// Define to the one symbol short name of this package.
#define PACKAGE_TARNAME @PACKAGE_TARNAME@

// Define to the version of this package.
#define PACKAGE_VERSION @PACKAGE_VERSION@

// Version number of package.
#define VERSION PACKAGE_VERSION

// Define to the address where bug reports for this package should be sent.
#define PACKAGE_BUGREPORT @PACKAGE_BUGREPORT@

// ---------------------------------------------------------------------------
// Path separator
#ifndef PATH_SEPARATOR
#  ifdef OS_WINDOWS
#    define PATH_SEPARATOR  '\\'
#  else
#    define PATH_SEPARATOR  '/'
#  endif
#endif

// ---------------------------------------------------------------------------
// Windows

// Always export symbols when compiling a shared library as this file is only
// included by internal modules when building the gflags library itself.
// The gflags_declare.h header file will set it to import these symbols otherwise.
#ifndef GFLAGS_DLL_DECL
#  if GFLAGS_IS_A_DLL && defined(_MSC_VER)
#    define GFLAGS_DLL_DECL __declspec(dllexport)
#  else
#    define GFLAGS_DLL_DECL
#  endif
#endif
// Flags defined by the gflags library itself must be exported
#ifndef GFLAGS_DLL_DEFINE_FLAG
#  define GFLAGS_DLL_DEFINE_FLAG GFLAGS_DLL_DECL
#endif

#ifdef OS_WINDOWS
// The unittests import the symbols of the shared gflags library
#  if GFLAGS_IS_A_DLL && defined(_MSC_VER)
#    define GFLAGS_DLL_DECL_FOR_UNITTESTS __declspec(dllimport)
#  endif
#  include "windows_port.h"
#endif


#endif // GFLAGS_CONFIG_H_
