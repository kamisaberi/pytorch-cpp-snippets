#include <torch/torch.h>
#include <opencv2/opencv.hpp>

int main() {
    // Create a sample 2D PyTorch tensor (for example, a 256x256 grayscale image)
    torch::Tensor tensor = torch::rand({256, 256}); // Random values in [0, 1]

    // Step 1: Move the tensor to CPU if it's on GPU
    if (tensor.is_cuda()) {
        tensor = tensor.cpu();
    }

    // Step 2: Access the data and create an OpenCV Mat
    // Convert tensor to a NumPy-compatible format (H, W)
    cv::Mat opencv_mat(tensor.size(0), tensor.size(1), CV_32F, tensor.data_ptr<float>());

    // Convert the Mat to CV_8U for displaying
    cv::Mat display_mat;
    opencv_mat.convertTo(display_mat, CV_8U, 255.0); // Scale to [0, 255]

    // Now you can use display_mat with OpenCV functions
    cv::imshow("Image", display_mat);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
