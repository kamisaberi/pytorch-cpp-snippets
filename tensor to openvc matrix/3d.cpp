#include <torch/torch.h>
#include <opencv2/opencv.hpp>

int main() {
    // Create a sample PyTorch tensor (3 channels, 256x256 image)
    torch::Tensor tensor = torch::rand({3, 256, 256});

    // Step 1: Move the tensor to CPU if it's on GPU
    if (tensor.is_cuda()) {
        tensor = tensor.cpu();
    }

    // Step 2: Access the data and create an OpenCV Mat
    // Note: We need to permute the dimensions for OpenCV (H, W, C)
    tensor = tensor.permute({1, 2, 0}); // Change from (C, H, W) to (H, W, C)

    // Step 3: Create OpenCV Mat from the tensor data
    cv::Mat opencv_mat(tensor.size(0), tensor.size(1), CV_32FC3, tensor.data_ptr<float>());

    // Convert the Mat to CV_8UC3 for displaying
    cv::Mat display_mat;
    opencv_mat.convertTo(display_mat, CV_8UC3, 255.0); // Scale to [0, 255]

    // Now you can use display_mat with OpenCV functions
    cv::imshow("Image", display_mat);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
