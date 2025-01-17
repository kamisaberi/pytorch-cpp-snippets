#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/transforms.h>

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

class MNISTDataset : public torch::data::Dataset<MNISTDataset> {
public:
    MNISTDataset(const std::string& root, bool train=true)
            : dataset(torch::data::datasets::MNIST(root, train).map(torch::data::transforms::Normalize<>(0.5, 0.5))) {
        dataset = dataset.map(Resize(28, 28)); // Resize to 28x28 (MNIST size)
    }

    torch::data::Example<> get(size_t index) override {
        return dataset.get(index);
    }

    size_t size() const override {
        return dataset.size();
    }

private:
    torch::data::datasets::MNIST dataset;
};

int main() {
    // Set the device
    torch::Device device(torch::kCUDA);

    // Load the MNIST dataset
    MNISTDataset mnist_dataset("/home/kami/Documents/ipynb/datasets/MNIST/raw/");
    auto data_loader = torch::data::make_data_loader(std::move(mnist_dataset), /*batch_size=*/64);

    // Iterate through the dataset
    for (auto& batch : *data_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        // Here you can implement your training loop
        std::cout << "Batch size: " << data.size(0) << std::endl;
    }

    return 0;
}
