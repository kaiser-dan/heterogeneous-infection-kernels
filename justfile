ROOT := `pwd`

# ----------------------
# List commands
default:
    @just --list --unsorted

# ----- Setup & Install -----
# Install according to uv.lock, with all extras and optional dependencies
[group('setup')]
install:
    uv sync --all-extras --all-groups

# Minimum install according to uv.lock
[group('setup')]
installmin:
    uv sync --no-dev

# Run all unit tests
[group('setup')]
test:
    uv run pytest


# ----- Research -----
# Render all Quarto documents
[group('research')]
renderall:
    find ./ -type f -name "*.qmd" | xargs -I{} quarto render {}


# ----- Cleanup -----
# Remove temporary and test files
[confirm]
[group('clean')]
clean:
    rm -rf results/
    find ./ -type f -name "*tmp" -delete

# Remove environment
[confirm]
[group('clean')]
rmenv:
    rm -rf .venv
