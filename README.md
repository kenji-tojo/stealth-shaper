# Stealth Shaper

This code uses C++20 and was tested on Mac and Ubuntu.

## How to use the code
Running the commands
```
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ cmake --build . -j
```
will create the executable ```stealth-headless``` under ```build/```


Then, the command
```
$ ./stealth-headless ../assets/bunny.obj
```
will run our stealth optimization. The result will be saved as ```build/result/stealth.obj```.

The program will also save the intermediate shapes under ```build/result```.