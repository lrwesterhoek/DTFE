#!/usr/bin/env bash
# Make the GUI launchable like any other application.
#
#   scripts/install_gui_app.sh               macOS: "DTFE Launcher.app" in /Applications (or
#                                            ~/Applications when /Applications is not writable)
#                                            + a shortcut to it on the Desktop.
#                                            Linux: a .desktop entry in ~/.local/share/applications
#                                            + a copy on the Desktop.
#   scripts/install_gui_app.sh --apps-dir D  put the app in D instead
#   scripts/install_gui_app.sh --no-desktop  no Desktop shortcut
#   scripts/install_gui_app.sh --remove      take both away again (macOS: the app goes to the Trash)
#
# The app only starts scripts/gui.sh of THIS repository, so it needs the GUI's PySide6 environment
# (scripts/install.sh sets it up) and must be re-made if the repository moves. Apps started from
# Finder get a bare PATH, so the launcher reads PATH and the DTFE settings (DTFE_*, TNG_API_KEY,
# PY, PYTHON) from the user's login shell, as a Terminal window would have them: the GUI then runs
# the same python3, wget and data root as the terminal. Its output goes to
# ~/Library/Logs/DTFE Launcher.log. Re-running replaces the app it made before (and only that).
#
# macOS privacy protection: an app started from Finder may not read iCloud Drive (where this
# repository lives) or an external drive (the T7) until it is given access, and for this app
# macOS does not ask -- it refuses. Give it Full Disk Access once (System Settings > Privacy &
# Security > Full Disk Access > +); the launcher explains this and opens that pane when it is
# refused. macOS gives that access to the PROCESS that is the app, so the bundle's executable is
# a small compiled program that stays alive as the parent of everything the GUI starts: with a
# shell script as the executable the process would be /bin/bash to macOS, which the access given
# to "DTFE Launcher" does not cover (tried: refused even with Full Disk Access switched on).
# The bundle is ad-hoc signed and rebuilt byte-identically, so the access given survives
# re-running this script (until the repository moves or the launcher itself changes).
#
# The GUI itself runs in a second, small bundle next to its PySide6 environment
# (~/.venvs/dtfe-gui/DTFE Launcher.app): an executable that embeds the environment's Python.
# macOS names a running app after the bundle of its process, and Python's own process belongs
# to Python.app -- so without it the Dock and Cmd-Tab say "Python". Started directly (e.g. kept
# in the Dock) it hands over to the real app above. Built for framework Pythons (Homebrew,
# python.org); scripts/gui.sh falls back to the plain interpreter when it is missing.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="DTFE Launcher"
BUNDLE_ID="local.dtfe.launcher"
APPS_DIR=""
DESKTOP=1
REMOVE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --apps-dir)   shift; APPS_DIR="${1:?--apps-dir needs a directory}" ;;
        --no-desktop) DESKTOP=0 ;;
        --remove)     REMOVE=1 ;;
        -h|--help)    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option '$1' (see --help)" >&2; exit 1 ;;
    esac
    shift
done

VENV="${DTFE_GUI_VENV:-$HOME/.venvs/dtfe-gui}"
GUI_PY="${DTFE_GUI_PYTHON:-$VENV/bin/python}"
need_gui_python() {
    "$GUI_PY" -c "import PySide6" >/dev/null 2>&1 \
        || { echo "PySide6 is not set up in $GUI_PY -- run scripts/install.sh first" >&2; exit 1; }
}

# ================================================================== Linux: a .desktop entry
if [ "$(uname -s)" != "Darwin" ]; then
    APPS="${APPS_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/applications}"
    ENTRY="$APPS/dtfe-launcher.desktop"
    DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")/dtfe-launcher.desktop"
    if [ "$REMOVE" -eq 1 ]; then
        rm -f "$ENTRY" "$DESK" "$APPS/dtfe-launcher.png"
        echo "removed the DTFE Launcher entries"; exit 0
    fi
    need_gui_python
    mkdir -p "$APPS"
    "$GUI_PY" "$REPO/python/gui/make_icon.py" "$APPS/dtfe-launcher.png"
    cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Name=$NAME
Comment=Run DTFE / PS-DTFE, the pipeline and the figures
Exec=env "DTFE_GUI_ICON=$APPS/dtfe-launcher.png" "$REPO/scripts/gui.sh"
Path=$REPO
Icon=$APPS/dtfe-launcher.png
Terminal=false
Categories=Science;Education;
EOF
    chmod +x "$ENTRY"
    if [ "$DESKTOP" -eq 1 ] && [ -d "$(dirname "$DESK")" ]; then
        cp "$ENTRY" "$DESK" && chmod +x "$DESK"
        # GNOME only launches Desktop entries it was told to trust
        command -v gio >/dev/null 2>&1 && gio set "$DESK" metadata::trusted true 2>/dev/null || true
    fi
    echo "installed: $ENTRY"
    exit 0
fi

# ================================================================== macOS: an app bundle
if [ -z "$APPS_DIR" ]; then
    if [ -w /Applications ]; then APPS_DIR=/Applications; else APPS_DIR="$HOME/Applications"; fi
fi
APP="$APPS_DIR/$NAME.app"
LINK="$HOME/Desktop/$NAME.app"
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister

HOST_ID="local.dtfe.launcher.gui"
HOST_APP="$VENV/$NAME.app"
ours() {   # an app bundle this script made (never touch someone else's app of the same name)
    [ -f "$1/Contents/Info.plist" ] \
        && [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$1/Contents/Info.plist" 2>/dev/null)" = "${2:-$BUNDLE_ID}" ]
}
to_trash() {
    local dest="$HOME/.Trash/$(basename "$1")"
    [ -e "$dest" ] && dest="$HOME/.Trash/$(basename "$1" .app) $(date +%H.%M.%S).app"
    mv "$1" "$dest"
}

if [ "$REMOVE" -eq 1 ]; then
    [ -L "$LINK" ] && rm "$LINK" && echo "removed the Desktop shortcut"
    if ours "$APP"; then
        "$LSREGISTER" -u "$APP" >/dev/null 2>&1 || true
        to_trash "$APP" && echo "moved $APP to the Trash"
    else
        echo "no $NAME.app of ours in $APPS_DIR"
    fi
    if ours "$HOST_APP" "$HOST_ID"; then rm -rf "$HOST_APP" && echo "removed the GUI host $HOST_APP"; fi
    exit 0
fi

need_gui_python
if [ -e "$APP" ] && ! ours "$APP"; then
    echo "$APP exists and was not made by this script; not replacing it" >&2
    exit 1
fi
mkdir -p "$APPS_DIR"
BUILD="$(mktemp -d -t dtfe-launcher)"
trap 'rm -rf "$BUILD"' EXIT
B="$BUILD/$NAME.app/Contents"
mkdir -p "$B/MacOS" "$B/Resources"

"$GUI_PY" "$REPO/python/gui/make_icon.py" "$B/Resources/AppIcon.png" --iconset "$BUILD/AppIcon.iconset"
iconutil -c icns "$BUILD/AppIcon.iconset" -o "$B/Resources/AppIcon.icns"

cat > "$B/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>                 <string>$NAME</string>
    <key>CFBundleDisplayName</key>          <string>$NAME</string>
    <key>CFBundleIdentifier</key>           <string>$BUNDLE_ID</string>
    <key>CFBundleExecutable</key>           <string>$NAME</string>
    <key>CFBundleIconFile</key>             <string>AppIcon</string>
    <key>CFBundlePackageType</key>          <string>APPL</string>
    <key>CFBundleShortVersionString</key>   <string>0.1</string>
    <key>CFBundleVersion</key>              <string>1</string>
    <key>LSApplicationCategoryType</key>    <string>public.app-category.education</string>
    <key>NSHighResolutionCapable</key>      <true/>
    <key>LSUIElement</key>                  <true/>
</dict>
</plist>
EOF
plutil -lint "$B/Info.plist" >/dev/null

# the launch script (Resources/launcher.sh): $REPO is baked in (quoted for the spaces in iCloud paths)
{
    echo '#!/bin/bash'
    echo "# $NAME -- made by scripts/install_gui_app.sh; re-run it if the repository moves."
    echo "# Run by the app's executable, which stays its parent (see install_gui_app.sh)."
    printf 'REPO=%q\n' "$REPO"
    cat <<'EOF'
LOG="$HOME/Library/Logs/DTFE Launcher.log"
mkdir -p "$(dirname "$LOG")"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') launch ===" >> "$LOG"
alert() {
    echo "launcher: $1" >> "$LOG"
    /usr/bin/osascript -e 'on run argv' -e 'display alert "DTFE Launcher" message (item 1 of argv) as critical' \
        -e 'end run' "$1" >/dev/null 2>&1
}
settings_alert() {   # the macOS privacy refusal: say why, offer the pane where access is given
    local a
    echo "launcher: no access to the repository (Full Disk Access?)" >> "$LOG"
    a="$(/usr/bin/osascript -e 'on run argv' \
        -e 'display alert "DTFE Launcher needs access to your files" message (item 1 of argv) buttons {"Cancel", "Open Settings"} default button 2' \
        -e 'end run' "$1" 2>/dev/null)"
    case "$a" in *"Open Settings"*) /usr/bin/open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles" ;; esac
}
if [ ! -x "$REPO/scripts/gui.sh" ]; then
    alert "The DTFE repository is not where this launcher expects it: $REPO -- run scripts/install_gui_app.sh from its new location."
    exit 1
fi
if ! head -c 1 "$REPO/scripts/gui.sh" >/dev/null 2>&1; then
    settings_alert "macOS has not given DTFE Launcher access to the DTFE repository ($REPO) -- iCloud Drive and external drives such as the T7 are protected. In System Settings > Privacy & Security > Full Disk Access, add DTFE Launcher (the + button, then Applications) and switch it on, then open DTFE Launcher again."
    exit 1
fi
# Finder starts apps with a bare PATH: take PATH and the DTFE settings from the login shell, as a
# Terminal window would have them. Capped at 15 s so a profile that waits for input cannot hang it.
SH="${SHELL:-/bin/zsh}"
ENVFILE="$(mktemp -t dtfe-launcher-env)"
"$SH" -ilc 'command env' </dev/null >"$ENVFILE" 2>/dev/null &
pid=$!
for _ in $(seq 150); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
kill "$pid" 2>/dev/null
while IFS= read -r line; do
    case "$line" in
        PATH=*|DTFE_*=*|TNG_API_KEY=*|PY=*|PYTHON=*) export "$line" ;;
    esac
done < "$ENVFILE"
rm -f "$ENVFILE"
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"      # in case the profile sets neither
export DTFE_GUI_ICON="$(cd "$(dirname "$0")" && pwd)/AppIcon.png"
VENV="${DTFE_GUI_VENV:-$HOME/.venvs/dtfe-gui}"
if ! "${DTFE_GUI_PYTHON:-$VENV/bin/python}" -c "import PySide6" >/dev/null 2>&1; then
    alert "The GUI's PySide6 environment ($VENV) is missing. Run scripts/install.sh in the DTFE repository."
    exit 1
fi
cd "$REPO"
exec "$REPO/scripts/gui.sh" "$@" >> "$LOG" 2>&1
EOF
} > "$B/Resources/launcher.sh"

# the executable: runs launcher.sh as its CHILD and waits, so it stays the process macOS asks
# about file access (the GUI and every job it starts inherit it). It also runs an (invisible) AppKit
# event loop: a second double-click while the GUI is open makes macOS "reopen" THIS app, which has no
# window, so it brings the GUI forward instead (before 2026-10-02 nothing appeared to happen).
command -v cc >/dev/null 2>&1 || { echo "no C compiler (xcode-select --install)" >&2; exit 1; }
cat > "$BUILD/launcher.m" <<'EOF'
#import <AppKit/AppKit.h>
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>

extern char **environ;
static pid_t child = 0;

static int exitCode(int status) { return WIFEXITED(status) ? WEXITSTATUS(status) : 1; }

@interface LauncherDelegate : NSObject <NSApplicationDelegate>
@end

@implementation LauncherDelegate
- (BOOL)applicationShouldHandleReopen:(NSApplication *)sender hasVisibleWindows:(BOOL)flag {
    (void)flag;
    /* the GUI is the child itself (launcher.sh and gui.sh exec into it); without that, the GUI
       host's bundle */
    NSRunningApplication *gui = [NSRunningApplication runningApplicationWithProcessIdentifier:child];
    if (gui == nil)
        gui = [[NSRunningApplication runningApplicationsWithBundleIdentifier:
                   [NSString stringWithUTF8String:HOST_ID]] firstObject];
    if (gui != nil) {
        if ([sender respondsToSelector:@selector(yieldActivationToApplication:)])
            [sender yieldActivationToApplication:gui];          /* macOS 14+: cooperative activation */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        [gui activateWithOptions:NSApplicationActivateAllWindows];
#pragma clang diagnostic pop
    }
    return NO;
}
@end

int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX], script[PATH_MAX + 32];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0 || realpath(exe, real) == NULL) return 1;
    snprintf(script, sizeof script, "%s/../Resources/launcher.sh", dirname(real));
    char **args = calloc((size_t)argc + 2, sizeof *args);
    if (args == NULL) return 1;
    args[0] = "/bin/bash";
    args[1] = script;
    for (int i = 1; i < argc; i++) args[i + 1] = argv[i];
    if (posix_spawn(&child, "/bin/bash", NULL, NULL, args, environ) != 0) return 1;

    /* leave with the child's status as soon as it ends: a watch on its exit, armed before the
       check below, so an end in between is caught by one or the other */
    dispatch_source_t ended = dispatch_source_create(DISPATCH_SOURCE_TYPE_PROC, (uintptr_t)child,
                                                     DISPATCH_PROC_EXIT, dispatch_get_main_queue());
    dispatch_source_set_event_handler(ended, ^{
        int status = 0;
        while (waitpid(child, &status, 0) < 0)
            if (errno != EINTR) exit(1);
        exit(exitCode(status));
    });
    dispatch_resume(ended);
    int status = 0;
    if (waitpid(child, &status, WNOHANG) == child) return exitCode(status);

    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        LauncherDelegate *delegate = [LauncherDelegate new];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
EOF
cc -O2 -Wall -Wextra -fobjc-arc -DHOST_ID="\"$HOST_ID\"" -framework AppKit \
    -o "$B/MacOS/$NAME" "$BUILD/launcher.m"

# ad-hoc signature: a stable identity for macOS's privacy settings (the same bytes give the same
# signature, so Full Disk Access given once survives a re-run of this script)
codesign --force --sign - "$BUILD/$NAME.app" >/dev/null 2>&1 || echo "note: could not sign the app (codesign)" >&2

# replace the previous version (ours, checked above: a bundle this script regenerates in full),
# register it for Spotlight / Launchpad
if [ -e "$APP" ]; then rm -rf "$APP"; fi
mv "$BUILD/$NAME.app" "$APP"
"$LSREGISTER" -f "$APP" >/dev/null 2>&1 || true
echo "installed: $APP"
echo "first launch: give it Full Disk Access (System Settings > Privacy & Security) -- macOS"
echo "protects iCloud Drive and external drives, and refuses instead of asking for this app"

# ---- the GUI host (see the header): the process the GUI runs in, named by our bundle
if [ "$GUI_PY" = "$VENV/bin/python" ]; then
    FWPREFIX="$("$GUI_PY" -c 'import sysconfig; print(sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX") or "")')"
    PYINC="$("$GUI_PY" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY") or "")')"
    if [ -z "$FWPREFIX" ] || [ ! -f "$PYINC/Python.h" ]; then
        echo "note: $GUI_PY is not a framework Python; the Dock and Cmd-Tab will say Python"
    elif [ -e "$HOST_APP" ] && ! ours "$HOST_APP" "$HOST_ID"; then
        echo "note: $HOST_APP exists and was not made by this script; GUI host left out" >&2
    else
        HB="$BUILD/host/$NAME.app"
        mkdir -p "$HB/Contents/MacOS" "$HB/Contents/Resources"
        cp "$APP/Contents/Resources/AppIcon.icns" "$HB/Contents/Resources/"
        sed -e "s|<string>$BUNDLE_ID</string>|<string>$HOST_ID</string>|" \
            -e '/<key>LSUIElement<\/key>/d' "$APP/Contents/Info.plist" > "$HB/Contents/Info.plist"
        plutil -lint "$HB/Contents/Info.plist" >/dev/null
        # the two paths as C string literals (they contain spaces; json quoting is valid C here)
        "$GUI_PY" -c 'import json, sys
print("#define VENV_PYTHON", json.dumps(sys.argv[1], ensure_ascii=False))
print("#define OUTER_APP", json.dumps(sys.argv[2], ensure_ascii=False))' "$GUI_PY" "$APP" > "$BUILD/host_paths.h"
        cat > "$BUILD/host.c" <<'EOF'
/* The GUI's process: the environment's Python, embedded, inside a bundle of our own, so that
   macOS names it "DTFE Launcher" (Dock, Cmd-Tab, menu bar). Made by scripts/install_gui_app.sh. */
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "host_paths.h"

int main(int argc, char **argv) {
    if (argc < 2 || strncmp(argv[1], "-psn_", 5) == 0) {
        /* started as an app (e.g. kept in the Dock): go through the real one, which sets up the
           environment and holds the file access */
        execl("/usr/bin/open", "open", "-a", OUTER_APP, (char *)NULL);
        return 1;
    }
    /* be the GUI environment's interpreter, exactly as its bin/python would be (CPython's own
       mechanism on macOS; it removes the variable again, so child processes do not see it) */
    setenv("__PYVENV_LAUNCHER__", VENV_PYTHON, 1);
    return Py_BytesMain(argc, argv);
}
EOF
        cc -O2 -Wall -Wextra -I"$PYINC" -I"$BUILD" -F"$FWPREFIX" -framework Python \
            -o "$HB/Contents/MacOS/$NAME" "$BUILD/host.c"
        codesign --force --sign - "$HB" >/dev/null 2>&1 || true
        if "$HB/Contents/MacOS/$NAME" -c "import PySide6" >/dev/null 2>&1; then
            rm -rf "$HOST_APP"
            mv "$HB" "$HOST_APP"
            echo "GUI host: $HOST_APP (the Dock and Cmd-Tab say $NAME)"
        else
            echo "note: the GUI host could not import PySide6; left out (the Dock will say Python)" >&2
        fi
    fi
fi

if [ "$DESKTOP" -eq 1 ]; then
    if [ -L "$LINK" ] || [ ! -e "$LINK" ]; then
        ln -sfn "$APP" "$LINK"
        echo "Desktop shortcut: $LINK"
    else
        echo "not making the Desktop shortcut: $LINK exists and is not a shortcut" >&2
    fi
fi
