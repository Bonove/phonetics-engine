import os
import pathlib

# On macOS with Homebrew, ctypes.util.find_library cannot locate espeak-ng
# because it relies on ldconfig which is Linux-only.  Point phonemizer at the
# dylib directly when running locally; on Linux/Docker the library is on the
# default search path so the env var is only set when the file actually exists.
_HOMEBREW_ESPEAK = pathlib.Path("/opt/homebrew/lib/libespeak-ng.dylib")
if _HOMEBREW_ESPEAK.exists() and "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(_HOMEBREW_ESPEAK)

# Fixtures will be added in later phases.
