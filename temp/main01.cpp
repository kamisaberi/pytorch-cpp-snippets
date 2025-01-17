#include <iostream>
#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/transforms.h>
#include <opencv2/opencv.hpp>

class Resize : public torch::data::transforms::Transform<torch::Tensor, torch::Tensor> {
public:
    Resize(int width, int height) : target_width(width), target_height(height) {}

    // Implement the apply method
    torch::Tensor apply(torch::Tensor input_tensor) {
        // Convert Torch tensor to OpenCV Mat
        cv::Mat input_image;
        if (input_tensor.ndimension() == 3) {
            input_image = cv::Mat(input_tensor.size(1), input_tensor.size(2), CV_32FC1, input_tensor.data_ptr<float>());
        } else {
            throw std::runtime_error("Input tensor must have 3 dimensions (C, H, W)");
        }

        // Resize the image
        cv::Mat resized_image;
        cv::resize(input_image, resized_image, cv::Size(target_width, target_height));

        // Convert back to Torch tensor
        return torch::from_blob(resized_image.data, {resized_image.rows, resized_image.cols, 1}, torch::kFloat32).permute({2, 0, 1}).clone();
    }

    // Override the operator() method
    torch::Tensor operator()(torch::Tensor input_tensor) {
        return apply(input_tensor);
    }

private:
    int target_width;
    int target_height;
};

int main() {

    // Define the MNIST dataset
    auto dataset = torch::data::datasets::MNIST("/home/kami/Documents/ipynb/datasets/MNIST/raw/")
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081)) // Normalize MNIST
            .map(torch::nn::functional::interpolate({32, 32},torch::kBilinear, false));

    // Create a data loader
    auto data_loader = torch::data::make_data_loader(std::move(dataset), /*batch_size=*/64);

    // Iterate through the dataset
//    for (auto &batch: *data_loader) {
//        // Get the images and labels
//        auto images = batch.data;
//        auto labels = batch.target;
//
//        // Example: Display the first image in the batch
//        cv::Mat image(28, 28, CV_32FC1, images[0].data_ptr<float>());
//        image.convertTo(image, CV_8UC1, 255.0); // Scale to [0, 255]
//        cv::imshow("Sample Image", image);
//        cv::waitKey(0); // Wait for a key press
//
//        // Process labels as needed
//        std::cout << "Label: " << labels[0].item<int>() << std::endl;
//
//        break; // Remove this break to iterate through the entire dataset
//    }

    return 0;
}
