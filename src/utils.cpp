#include "utils.hpp"

namespace utils
{
    int64_t get_num_parameters(torch::nn::Module &model){
        int64_t total_params = 0;
        for(const auto &parameters: model.parameters()){
            total_params+=parameters.numel();
        }
        return total_params;
    }
    
    bool is_corrupt(const std::string& path) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Corrupt or unreadable image: " << path << std::endl;
            return true;
        }
        return false;
    }

    void verify_corrupt_images_dir(fs::path& path){
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (fs::is_regular_file(entry.path())) {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                    is_corrupt(entry.path().string());
                }
            }
        }
    }

    torch::Tensor compute_class_weights(const std::vector<int>& labels) {
        if (labels.empty()) {
            throw std::runtime_error("Label vector is empty.");
        }
        
        std::unordered_map<int, int> class_counts;
        int total_samples = labels.size();
        
        // Count number of samples per class
        for (int label : labels) {
            class_counts[label]++;
        }
        
        int min_label = std::numeric_limits<int>::max();
        int max_label = std::numeric_limits<int>::min();
        for (const auto& pair : class_counts) {
            min_label = std::min(min_label, pair.first);
            max_label = std::max(max_label, pair.first);
        }
        
        int num_classes = max_label - min_label + 1;
        std::vector<float> weights(num_classes, 0.0f);
        
        // Compute balanced class weights
        for (const auto& [class_label, count] : class_counts) {
            float weight = static_cast<float>(total_samples) / (class_counts.size() * count);
            weights[class_label - min_label] = weight;
        }
        
        return torch::tensor(weights);
    }

    torch::Tensor unnormalize(const torch::Tensor& img) {
        auto mean = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
        auto std = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
        return img.mul(std).add(mean).clamp(0, 1);
    }
    void visualizeBatchImage(const torch::Tensor& batch_images, const torch::Tensor& batch_labels, int index) {
        torch::Tensor img = batch_images[index].detach().cpu();
        img = unnormalize(img);  // Unnormalize
        img = img.permute({1, 2, 0}).mul(255).to(torch::kU8);

        int height = img.size(0);
        int width = img.size(1);
        cv::Mat cv_image(cv::Size(width, height), CV_8UC3, img.data_ptr());
    
        std::memcpy(cv_image.data, img.data_ptr(), 
                    sizeof(torch::kByte) * img.numel());
        int label = batch_labels[index].item<int>();
        // Convert RGB to BGR
        cv::cvtColor(cv_image, cv_image, cv::COLOR_RGB2BGR);
        std::cout << "Target label : " << std::to_string(label) << std::endl;
        // Display
        cv::imshow("Batch Image", cv_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::tuple<float, float, float> compute_batch_metrics_binary(const std::vector<int>& targets, const std::vector<int>& preds) {
        int64_t tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < preds.size(); ++i) {
            bool p = preds[i] == 1;
            bool t = targets[i] == 1;
    
            if (p && t) tp++;
            else if (p && !t) fp++;
            else if (!p && t) fn++;
        }
    
        float precision = tp / float(tp + fp + 1e-6);
        float recall = tp / float(tp + fn + 1e-6);
        float f1 = 2 * precision * recall / (precision + recall + 1e-6);
    
        return std::make_tuple(precision, recall, f1);
    }
    

    std::tuple<float, float, float> compute_batch_metrics_multiclass(const std::vector<int>& targets, const std::vector<int>& preds, int num_classes){
        std::vector<int> tp(num_classes, 0), fp(num_classes, 0), fn(num_classes, 0);
    
        for (size_t i = 0; i < preds.size(); ++i) {
            int p = preds[i];
            int t = targets[i];
            if (p == t) {
                tp[p]++;
            } else {
                fp[p]++;
                fn[t]++;
            }
        }
    
        float total_precision = 0.0f, total_recall = 0.0f, total_f1 = 0.0f;
        int non_zero_classes = 0;
    
        for (int i = 0; i < num_classes; ++i) {
            float precision = tp[i] / float(tp[i] + fp[i] + 1e-6);
            float recall = tp[i] / float(tp[i] + fn[i] + 1e-6);
            float f1 = 2 * precision * recall / (precision + recall + 1e-6);
    
            if ((tp[i] + fp[i] + fn[i]) > 0) { // class exists in batch
                total_precision += precision;
                total_recall += recall;
                total_f1 += f1;
                non_zero_classes++;
            }
        }
    
        // Macro average for batch
        return {
            total_precision / non_zero_classes,
            total_recall / non_zero_classes,
            total_f1 / non_zero_classes
        };
    }

    std::tuple<std::shared_ptr<std::vector<std::string>>, std::shared_ptr<std::vector<int>> > get_image_path_and_labels(fs::path image_dir, fs::path annotation_path, bool binary){
        std::vector<std::string> image_paths; 
        std::vector<int> labels;

        if (!fs::exists(image_dir) || !fs::is_directory(image_dir)) {
            throw std::runtime_error("Invalid directory path: " + image_dir.string());
        }
        if (!fs::exists(annotation_path) || !fs::is_regular_file(annotation_path)) {
            throw std::runtime_error("Invalid annotation file path: " + annotation_path.string());
        }

        std::unordered_map<std::string, int> image_to_label;
        std::ifstream infile(annotation_path.string());
        std::string line;

        while (std::getline(infile, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string image_name;
            int class_id, species_id, _; // the 4th column is ignored
            if (!(iss >> image_name >> class_id >> species_id >> _)) continue; // Ensure that we get 4 valid column like lines before provessing 

            int label = binary ? (std::isupper(image_name[0]) ? 1 : 0) : class_id; // if binary then based on this : 
            // All images with 1st letter as captial are cat images while images with small first letter are dog images. Cat 1 Dog 0
            // Else we retrieve the class ID // https://www.kaggle.com/datasets/julinmaloof/the-oxfordiiit-pet-dataset/data 
            image_to_label[image_name] = label;
        }

        std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".png"};
        for (const auto& entry : fs::directory_iterator(image_dir)) {
            if (fs::is_regular_file(entry)) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (std::find(valid_extensions.begin(), valid_extensions.end(), ext) == valid_extensions.end())
                    continue;

                std::string filename = entry.path().stem().string(); // used as key ex : Bombay_93
                if (image_to_label.find(filename) != image_to_label.end()) {
                    image_paths.push_back(entry.path().string());
                    labels.push_back(image_to_label[filename]);
                }
            }
        }
        return std::make_tuple(std::make_shared<std::vector<std::string>>(std::move(image_paths)),
               std::make_shared<std::vector<int>>(std::move(labels)));
    }
    std::tuple<std::shared_ptr<std::vector<std::string>>, std::shared_ptr<std::vector<std::string>>, 
               std::shared_ptr<std::vector<int>>, std::shared_ptr<std::vector<int>> > train_test_split(std::shared_ptr<std::vector<std::string>> image_paths, std::shared_ptr<std::vector<int>> labels  ,float test_size)
    {
        if (image_paths->size() != labels->size()) {
            throw std::invalid_argument("image_paths and labels must have the same size.");
        }

        size_t total_size = image_paths->size();
        size_t test_count = static_cast<size_t>(total_size * test_size);
        std::vector<size_t> indices(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            indices[i] = i;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);

        // Preallocate
        auto train_images = std::make_shared<std::vector<std::string>>();
        auto test_images = std::make_shared<std::vector<std::string>>();
        auto train_labels = std::make_shared<std::vector<int>>();
        auto test_labels = std::make_shared<std::vector<int>>();

        train_images->reserve(total_size - test_count);
        test_images->reserve(test_count);
        train_labels->reserve(total_size - test_count);
        test_labels->reserve(test_count);

        for (size_t i = 0; i < total_size; ++i) {
            size_t idx = indices[i];
            if (i < test_count) {
                test_images->emplace_back(std::move((*image_paths)[idx])); // since we are moving the values from image paths to test_images the image_paths are empty and do not contain values
                test_labels->emplace_back((*labels)[idx]);
            } else {
                train_images->emplace_back(std::move((*image_paths)[idx]));
                train_labels->emplace_back((*labels)[idx]);
            }
        }

        return {train_images, test_images, train_labels, test_labels};
    }
    
    cv::Mat TensortoCV(torch::Tensor& tensor_image) {
        // std::cout << "Tensor shape: " << tensor_image.sizes() << std::endl;
        tensor_image = tensor_image.to(torch::kCPU);
        tensor_image = tensor_image.permute({1, 2, 0}).mul(255).clamp(0, 255).to(torch::kU8); // CHW -> HWC
        tensor_image = tensor_image.contiguous().to(torch::kCPU);
        // std::cout << "tensor to cv done" << std::endl;

        return cv::Mat(tensor_image.size(0), tensor_image.size(1), CV_8UC3, tensor_image.data_ptr<uint8_t>());
    }
    torch::Tensor CVtoTensor(cv::Mat& image) {
        // Convert OpenCV Mat to Torch Tensor
        image.convertTo(image, CV_32F, 1.0 / 255);
        torch::Tensor tensor_image = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kFloat);
        tensor_image = tensor_image.permute({2, 0, 1}); // Change from HWC to CHW
        return tensor_image.clone(); 
    }
    

    ModelArgs parseArguments(int argc, char* argv[]) {
        ModelArgs args;
        args.binary_mode = true;                // Default to binary mode
        
        // Print usage if there's no enough arguments
        if (argc < 2) {
            std::cerr << "Usage:" << std::endl;
            std::cerr << "  " << argv[0] << " dataset <mnist_data_path> <save_path>" << std::endl;
            std::cerr << "  " << argv[0] << " image <image_path> <label_path> <save_path> [--net=<cnn|convnext|mobilevit>] [--binary]" << std::endl;
            exit(1);
        }
        std::string mode = argv[1];
        // DATASET MODEL:
        if (mode == "dataset") {
            if (argc < 4) {
                std::cerr << "Usage for dataset mode: " << argv[0] << " dataset <mnist_data_path> <save_path>" << std::endl;
                exit(1);
            }
            
            args.model_type = ModelArgs::DATASET_MODEL;
            
            fs::path arg_path(argv[2]);
            if (arg_path.is_absolute()) {
                args.dataset_path = arg_path;
            } else {
                args.dataset_path = fs::current_path() / arg_path;
            }
            
            fs::path save_arg_path(argv[3]);
            if (save_arg_path.is_absolute()) {
                args.save_path = save_arg_path;
            } else {
                args.save_path = fs::current_path() / save_arg_path;
            }
            
            std::cout << "Mode: Dataset Model" << std::endl;
            std::cout << "Dataset Path: " << args.dataset_path << std::endl;
            std::cout << "Save Path: " << args.save_path << std::endl;
            
            // Validate paths
            if (!fs::exists(args.dataset_path)) {
                std::cerr << "Error: Dataset path does not exist: " << args.dataset_path << std::endl;
                exit(1);
            }
        }
        // IMAGE+LABEL MODEL: requires 3 additional args (image path, label path, save path)
        else if (mode == "image") {
            if (argc < 5) {
                std::cerr << "Usage for image mode: " << argv[0] << " image <image_path> <label_path> <save_path>" << std::endl;
                exit(1);
            }
            
            args.model_type = ModelArgs::IMAGE_LABEL_MODEL;

            fs::path img_path(argv[2]);
            if (img_path.is_absolute()) {
                args.image_path = img_path;
            } else {
                args.image_path = fs::current_path() / img_path;
            }

            fs::path lbl_path(argv[3]);
            if (lbl_path.is_absolute()) {
                args.label_path = lbl_path;
            } else {
                args.label_path = fs::current_path() / lbl_path;
            }

            fs::path save_path(argv[4]);
            if (save_path.is_absolute()) {
                args.save_path = save_path;
            } else {
                args.save_path = fs::current_path() / save_path;
            }
            
            std::cout << "Mode: Oxford Dataset" << std::endl;
            std::cout << "Image Path: " << args.image_path << std::endl;
            std::cout << "Label Path: " << args.label_path << std::endl;
            std::cout << "Save Path: " << args.save_path << std::endl;
            
            // Check for optional network type and binary mode flags
            for (int i = 5; i < argc; i++) {
                std::string arg = argv[i];
                
                // Check for binary mode flag
                if (arg == "--binary") {
                    args.binary_mode = true;
                    std::cout << "Binary Mode: Enabled" << std::endl;
                    continue;
                }
                
                // Check for neural network type
                if (arg.substr(0, 6) == "--net=") {
                    std::string net_type = arg.substr(6);
                    if (net_type == "cnn") {
                        args.net_type = ModelArgs::CNN;
                        std::cout << "Neural Network: CNN" << std::endl;
                    } else if (net_type == "convnext") {
                        args.net_type = ModelArgs::CONVNEXT;
                        std::cout << "Neural Network: ConvNeXt" << std::endl;
                    } else if (net_type == "mobilevit") {
                        args.net_type = ModelArgs::MOBILEVIT;
                        std::cout << "Neural Network: MobileViT" << std::endl;
                    } else {
                        std::cerr << "Warning: Unknown network type '" << net_type << "'. Using default." << std::endl;
                    }
                }
            }
            
            // Validate paths
            if (!fs::exists(args.image_path)) {
                std::cerr << "Error: Image path does not exist: " << args.image_path << std::endl;
                exit(1);
            }
            
            if (!fs::exists(args.label_path)) {
                std::cerr << "Error: Label path does not exist: " << args.label_path << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << "Invalid mode: " << mode << std::endl;
            std::cerr << "Usage:" << std::endl;
            std::cerr << "  " << argv[0] << " dataset <mnist_data_path> <save_path> " << std::endl;
            std::cerr << "  " << argv[0] << " image <image_path> <label_path> <save_path> [--net=<cnn|convnext|mobilevit>] [--binary]" << std::endl;
            exit(1);
        }
        
        return args;
    }
} // namespace utils

