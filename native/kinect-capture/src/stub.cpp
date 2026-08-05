#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--capabilities") {
            std::cout << "[]\n";
            return 0;
        }
    }
    std::cerr << "kinect2-capture-worker: Kinect v2 support was not compiled; install Kinect for Windows SDK 2.0 and rebuild\n";
    return 1;
}
