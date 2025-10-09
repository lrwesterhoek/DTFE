# Cross-Platform Build Instructions for DTFE

This project now supports building on macOS, Linux, and Windows with automatic platform detection.

## Quick Start

1. **Test your platform detection:**
   ```bash
   make test-platform
   ```

2. **Build the main executable:**
   ```bash
   make DTFE
   ```

3. **Build the shared library:**
   ```bash
   make library
   ```

4. **Clean build files:**
   ```bash
   make clean
   ```

## Platform-Specific Setup

### macOS
- **Prerequisites:** Xcode Command Line Tools
- **Package Manager:** Homebrew (recommended)
- **Install dependencies:**
  ```bash
  brew install gsl boost cgal mpfr hdf5 gmp llvm
  ```
- The Makefile automatically detects Homebrew paths

### Linux (Ubuntu/Debian)
- **Install dependencies:**
  ```bash
  sudo apt-get update
  sudo apt-get install build-essential
  sudo apt-get install libgsl-dev libboost-all-dev libcgal-dev 
  sudo apt-get install libmpfr-dev libhdf5-dev libgmp-dev
  ```
- For other distributions, use equivalent package manager commands

### Linux (Red Hat/CentOS/Fedora)
- **Install dependencies:**
  ```bash
  sudo yum groupinstall "Development Tools"  # or dnf on newer systems
  sudo yum install gsl-devel boost-devel CGAL-devel
  sudo yum install mpfr-devel hdf5-devel gmp-devel
  ```

### Windows Option 1: MSVC (Visual Studio)
1. **Install Visual Studio:** Download Visual Studio Community (free)
2. **Install vcpkg package manager:**
   ```cmd
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   ```
3. **Install dependencies via vcpkg:**
   ```cmd
   vcpkg install gsl:x64-windows boost:x64-windows cgal[core]:x64-windows
   vcpkg install mpfr:x64-windows hdf5:x64-windows gmp:x64-windows
   ```
4. **Set environment variable:**
   ```cmd
   set VCPKG_ROOT=C:\path\to\vcpkg\installed\x64-windows
   ```
5. **Open Developer Command Prompt and build:**
   ```cmd
   nmake DTFE
   # or if GNU make is available:
   make DTFE
   ```

### Windows Option 2: MSYS2/MinGW-w64
1. **Install MSYS2:** Download from https://www.msys2.org/
2. **Open MSYS2 terminal and install dependencies:**
   ```bash
   # Update package database
   pacman -Syu
   
   # Install build tools
   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make
   
   # Install libraries
   pacman -S mingw-w64-x86_64-gsl mingw-w64-x86_64-boost
   pacman -S mingw-w64-x86_64-cgal mingw-w64-x86_64-mpfr
   pacman -S mingw-w64-x86_64-hdf5 mingw-w64-x86_64-gmp
   ```
3. **Use MinGW64 terminal for building**

## Customization

### Override Library Paths
If libraries are installed in non-standard locations:

```bash
# Method 1: Command line
make DTFE GSL_PATH_OVERRIDE=/custom/path/to/gsl

# Method 2: Environment variables
export GSL_PATH_OVERRIDE=/custom/path/to/gsl
export BOOST_PATH_OVERRIDE=/custom/path/to/boost
make DTFE
```

### Override Compiler
```bash
# Use a specific compiler
make DTFE CC_OVERRIDE=g++-11

# Or set as environment variable
export CC_OVERRIDE=clang++
make DTFE
```

## Troubleshooting

### Common Issues

1. **Compiler not found:**
   - Install build tools for your platform
   - Use `CC_OVERRIDE` to specify exact compiler path

2. **Libraries not found:**
   - Check if development packages are installed
   - Use `*_PATH_OVERRIDE` variables to specify custom paths
   - Run `make test-platform` to verify detected paths

3. **OpenMP issues:**
   - On macOS: `brew install libomp`
   - On Windows: Ensure MinGW version supports OpenMP

4. **Permission errors (Linux/macOS):**
   - Don't run make with sudo unless necessary
   - Ensure you have write permissions to build directories

### Platform Detection Issues
If the automatic platform detection fails:
1. Check `make test-platform` output
2. Manually set platform-specific variables
3. Report the issue with your system details

## Advanced Configuration

### Build Types
The Makefile supports different build configurations through the `OPTIONS` variable:

- **Spatial dimensions:** `-DNO_DIM=2` or `-DNO_DIM=3`
- **Precision:** Add `-DDOUBLE` for double precision
- **Features:** Various `-D` flags for enabling/disabling features

### Custom Build Flags
```bash
# Add custom compiler flags
make DTFE EXTRA_CFLAGS="-march=native -mtune=native"
```

## Testing Cross-Platform Compatibility

Several testing methods are available to validate the cross-platform Makefile:

### 1. Built-in Test Suite
```bash
# Run comprehensive test suite
make test-suite

# Test current platform only
make test-platform

# Test all platform configurations (simulated)
make test-all-platforms

# Validate specific components
make test-validation
```

### 2. Platform-Specific Tests
```bash
# Test Windows MinGW compatibility
./test-windows-compat.sh

# Test Windows MSVC compatibility
./test-msvc-compat.sh

# Test Linux with Docker (requires Docker)
./test-docker-linux.sh
```

### 3. Manual Testing
To manually test a specific platform configuration:
```bash
# Simulate Linux build
make test-platform UNAME_S=Linux

# Simulate Windows MinGW build
make test-platform UNAME_S=Windows WINDOWS_COMPILER=mingw

# Simulate Windows MSVC build
make test-platform UNAME_S=Windows WINDOWS_COMPILER=msvc

# Test with custom paths
make test-platform UNAME_S=Linux GSL_PATH_OVERRIDE=/custom/path

# Test MSVC with vcpkg
make test-platform UNAME_S=Windows WINDOWS_COMPILER=msvc VCPKG_ROOT=C:/vcpkg/installed/x64-windows
```
