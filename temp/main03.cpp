#include <torch/torch.h>
#include <iostream>
#include <torchvision/datasets/mnist.h>
#include <torchvision/transforms.h>
#include <opencv2/opencv.hpp> // Include OpenCV for resizing

// Custom resize function
torch::Tensor resize(const torch::Tensor& img, int height, int width) {
    cv::Mat img_mat = torch::data::utils::toMat(img); // Convert torch tensor to OpenCV Mat
    cv::Mat resized_img;
    cv::resize(img_mat, resized_img, cv::Size(width, height)); // Resize using OpenCV
    return torch::from_blob(resized_img.data, {1, height, width, 1}, torch::kByte).clone(); // Convert back to tensor
}

int main() {
    // Set the root directory for the dataset
    std::string root = "./data";

    // Define the transformations
    auto transform = torch::data::transforms::Compose({
                                                              torch::data::transforms::Lambda([=](torch::Tensor img) {
                                                                  return resize(img, 32, 32); // Resize to 32x32
                                                              }),
                                                              torch::data::transforms::ToTensor(),
                                                              torch::data::transforms::Normalize({0.1307}, {0.3081})
                                                      });

    // Load the MNIST training dataset
    auto train_dataset = torchvision::datasets::MNIST(root, /* train */ true, transform, /* download */ true);

    // Output the size of the dataset
    std::cout << "Number of training samples: " << train_dataset.size().value() << std::endl;

    return 0;
}
