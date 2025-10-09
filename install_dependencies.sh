# DTFE Dependency Installation Script
# 
# Automatically installs required dependencies for building DTFE
# Supports: macOS (Intel/Apple Silicon), Ubuntu, Debian, Fedora, RHEL, CentOS, Arch, Manjaro
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

# Detect OS and distribution
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        # Detect architecture
        ARCH=$(uname -m)
        if [[ "$ARCH" == "arm64" ]]; then
            print_info "Detected: macOS (Apple Silicon)"
        else
            print_info "Detected: macOS (Intel)"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        ARCH=$(uname -m)
        
        # Detect Linux distribution
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            DISTRO=$ID
            DISTRO_LIKE=$ID_LIKE
            DISTRO_VERSION=$VERSION_ID
            print_info "Detected: $PRETTY_NAME ($ARCH)"
        else
            print_error "Cannot detect Linux distribution"
            exit 1
        fi
    else
        print_error "Unsupported operating system: $OSTYPE"
        print_info "This script supports macOS and Linux only"
        exit 1
    fi
}

# Check if running as root (we don't want this)
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "This script should NOT be run as root/sudo"
        print_info "The script will prompt for sudo password when needed"
        read -p "Continue anyway? (not recommended) [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Install Homebrew on macOS
install_homebrew() {
    if command -v brew &> /dev/null; then
        print_success "Homebrew is already installed"
        return 0
    fi
    
    print_info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon
    if [[ "$ARCH" == "arm64" ]]; then
        if [[ -f /opt/homebrew/bin/brew ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    fi
    
    print_success "Homebrew installed successfully"
}

# Install dependencies on macOS
install_macos() {
    print_header "Installing macOS Dependencies"
    
    # Check for Xcode Command Line Tools
    if ! xcode-select -p &> /dev/null; then
        print_info "Installing Xcode Command Line Tools..."
        xcode-select --install
        print_warning "Please complete the Xcode Command Line Tools installation"
        print_warning "Then run this script again"
        exit 0
    else
        print_success "Xcode Command Line Tools already installed"
    fi
    
    # Install Homebrew if needed
    install_homebrew
    
    # Update Homebrew
    print_info "Updating Homebrew..."
    brew update
    
    # Install dependencies
    print_info "Installing DTFE dependencies..."
    
    PACKAGES=(gsl boost cgal mpfr hdf5 gmp)
    
    for package in "${PACKAGES[@]}"; do
        if brew list "$package" &> /dev/null; then
            print_success "$package is already installed"
        else
            print_info "Installing $package..."
            brew install "$package"
        fi
    done
    
    print_success "All dependencies installed successfully!"
}

# Install dependencies on Ubuntu/Debian
install_ubuntu_debian() {
    print_header "Installing Ubuntu/Debian Dependencies"
    
    print_info "Updating package lists..."
    sudo apt-get update
    
    print_info "Installing build tools..."
    sudo apt-get install -y build-essential
    
    print_info "Installing DTFE dependencies..."
    sudo apt-get install -y \
        libgsl-dev \
        libboost-all-dev \
        libcgal-dev \
        libmpfr-dev \
        libhdf5-dev \
        libgmp-dev
    
    print_success "All dependencies installed successfully!"
}

# Install dependencies on Fedora/RHEL/CentOS
install_fedora_rhel() {
    print_header "Installing Fedora/RHEL/CentOS Dependencies"
    
    # Determine package manager (dnf for newer, yum for older)
    if command -v dnf &> /dev/null; then
        PKG_MGR="dnf"
        # Check if it's dnf5 (which has different syntax)
        DNF_VERSION=$(dnf --version 2>/dev/null | head -n1 | grep -oP '(?<=dnf5 )\d+' || echo "")
        if [[ -n "$DNF_VERSION" ]]; then
            print_info "Using package manager: dnf5"
            IS_DNF5=true
        else
            print_info "Using package manager: dnf4"
            IS_DNF5=false
        fi
    elif command -v yum &> /dev/null; then
        PKG_MGR="yum"
        IS_DNF5=false
        print_info "Using package manager: yum"
    else
        print_error "Neither dnf nor yum found"
        exit 1
    fi
    
    print_info "Installing development tools..."
    if [[ "$IS_DNF5" == true ]]; then
        # dnf5 uses "group install" instead of "groupinstall"
        sudo $PKG_MGR group install -y "Development Tools" || \
        sudo $PKG_MGR install -y @development-tools || \
        sudo $PKG_MGR install -y gcc gcc-c++ make automake autoconf libtool
    else
        # dnf4 and yum use "groupinstall"
        sudo $PKG_MGR groupinstall -y "Development Tools" || \
        sudo $PKG_MGR install -y @development-tools || \
        sudo $PKG_MGR install -y gcc gcc-c++ make automake autoconf libtool
    fi
    
    print_info "Installing DTFE dependencies..."
    sudo $PKG_MGR install -y \
        gsl-devel \
        boost-devel \
        CGAL-devel \
        mpfr-devel \
        hdf5-devel \
        gmp-devel
    
    print_success "All dependencies installed successfully!"
}

# Install dependencies on Arch/Manjaro
install_arch() {
    print_header "Installing Arch/Manjaro Dependencies"
    
    print_info "Updating package database..."
    sudo pacman -Sy
    
    print_info "Installing build tools and dependencies..."
    sudo pacman -S --needed --noconfirm \
        base-devel \
        gsl \
        boost \
        cgal \
        mpfr \
        hdf5 \
        gmp
    
    print_success "All dependencies installed successfully!"
}

# Install dependencies on openSUSE
install_opensuse() {
    print_header "Installing openSUSE Dependencies"
    
    print_info "Installing development tools and dependencies..."
    sudo zypper install -y -t pattern devel_C_C++
    sudo zypper install -y \
        gsl-devel \
        boost-devel \
        cgal-devel \
        mpfr-devel \
        hdf5-devel \
        gmp-devel
    
    print_success "All dependencies installed successfully!"
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    local all_good=true
    
    # Check compiler
    if command -v g++ &> /dev/null || command -v clang++ &> /dev/null; then
        print_success "C++ compiler found"
    else
        print_error "No C++ compiler found"
        all_good=false
    fi
    
    # Check make
    if command -v make &> /dev/null; then
        print_success "Make utility found"
    else
        print_error "Make utility not found"
        all_good=false
    fi
    
    # For macOS, check specific packages
    if [[ "$OS" == "macos" ]]; then
        for package in gsl boost cgal mpfr hdf5 gmp; do
            if brew list "$package" &> /dev/null 2>&1; then
                print_success "$package is installed"
            else
                print_warning "$package might not be installed correctly"
            fi
        done
    fi
    
    if $all_good; then
        print_success "All verification checks passed!"
        return 0
    else
        print_warning "Some verification checks failed"
        return 1
    fi
}

# Print next steps
print_next_steps() {
    print_header "Next Steps"
    
    echo "1. Test the build system:"
    echo -e "   ${GREEN}make test-platform${NC}"
    echo ""
    echo "2. Build DTFE:"
    echo -e "   ${GREEN}make DTFE${NC}"
    echo ""
    echo "3. Or build in debug mode:"
    echo -e "   ${GREEN}make DTFE BUILD_MODE=debug${NC}"
    echo ""
    echo "4. For more information, see:"
    echo -e "   ${BLUE}README.md${NC}"
    echo ""
}

# Main installation function
main() {
    print_header "DTFE Dependency Installer"
    
    # Detect OS
    detect_os
    
    # Check if running as root
    check_root
    
    # Install based on OS/distribution
    case "$OS" in
        macos)
            install_macos
            ;;
        linux)
            # Check both DISTRO and DISTRO_LIKE for better compatibility
            # Normalize to lowercase for comparison
            DISTRO_LOWER=$(echo "$DISTRO" | tr '[:upper:]' '[:lower:]')
            DISTRO_LIKE_LOWER=$(echo "$DISTRO_LIKE" | tr '[:upper:]' '[:lower:]')
            
            # Determine which installer to use
            INSTALLER=""
            
            # Check for Debian-based distros
            if [[ "$DISTRO_LOWER" =~ ^(ubuntu|debian|pop|linuxmint|elementary|zorin|kali|parrot|raspbian)$ ]] || \
               [[ "$DISTRO_LIKE_LOWER" =~ (ubuntu|debian) ]]; then
                INSTALLER="ubuntu_debian"
            
            # Check for Fedora-based distros (including Asahi Fedora)
            elif [[ "$DISTRO_LOWER" =~ ^(fedora|rhel|centos|rocky|almalinux|scientific|oracle|nobara|ultramarine|asahi)$ ]] || \
                 [[ "$DISTRO_LOWER" =~ fedora ]] || \
                 [[ "$DISTRO_LIKE_LOWER" =~ (fedora|rhel|centos) ]]; then
                INSTALLER="fedora_rhel"
            
            # Check for Arch-based distros
            elif [[ "$DISTRO_LOWER" =~ ^(arch|manjaro|endeavouros|garuda|artix|parabola|cachyos)$ ]] || \
                 [[ "$DISTRO_LIKE_LOWER" =~ arch ]]; then
                INSTALLER="arch"
            
            # Check for openSUSE-based distros
            elif [[ "$DISTRO_LOWER" =~ ^(opensuse|sles|suse)$ ]] || \
                 [[ "$DISTRO_LIKE_LOWER" =~ (opensuse|suse) ]]; then
                INSTALLER="opensuse"
            fi
            
            # Run the appropriate installer
            if [[ -n "$INSTALLER" ]]; then
                case "$INSTALLER" in
                    ubuntu_debian)
                        install_ubuntu_debian
                        ;;
                    fedora_rhel)
                        install_fedora_rhel
                        ;;
                    arch)
                        install_arch
                        ;;
                    opensuse)
                        install_opensuse
                        ;;
                esac
            else
                print_error "Unsupported Linux distribution: $DISTRO"
                print_info "Detected ID: $DISTRO"
                print_info "Detected ID_LIKE: $DISTRO_LIKE"
                print_info "Please install dependencies manually (see README.md)"
                exit 1
            fi
            ;;
    esac
    
    # Verify installation
    echo ""
    verify_installation
    
    # Print next steps
    echo ""
    print_next_steps
}

# Run main function
main