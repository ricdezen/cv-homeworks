# De Zen Riccardo - Homework 3

For the third homework I'm submitting my code and report for lab experience 5. To compile and run the code cmake should
be enough.

I have commented out a line in `CMakeLists.txt` because it might not be needed on all machines depending on platform and
environment variables. It was needed on my main pc.

The program accepts command line arguments to specify data paths. A help (`-h` or `--help`) option is available for the
program.

I tested my code on all provided datasets. Also tried a few pictures taken with my own phone, but I did not find the
results interesting.

**Tools**:

- Windows 10 Home.
- OpenCV 4.5.2 compiled with MinGW (Debug build, if that makes any difference).
- CLion 2021.1.1.

**Example usage**

```bash
# (These are also the default values)
# Files are in ./lab5_data/lab
# Images have .bmp suffix.
# Direction is to the right.
# Fov is 66°.
lab5 -p ./lab5_data/lab -s bmp -d r -f 66
```