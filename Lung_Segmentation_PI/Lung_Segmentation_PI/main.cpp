#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;

// Função para carregar imagens e máscaras de uma pasta e garantir que estão pareadas
void load_images_and_masks(const std::string& images_folder, const std::string& masks_folder, std::vector<cv::Mat>& images, std::vector<cv::Mat>& masks) {
    std::vector<std::string> images_filenames;
    std::vector<std::string> masks_filenames;

    // Obter lista de arquivos de imagens e máscaras
    for (const auto& entry : fs::directory_iterator(images_folder))
        images_filenames.push_back(entry.path().string());

    for (const auto& entry : fs::directory_iterator(masks_folder))
        masks_filenames.push_back(entry.path().string());

    // Ordenar listas de arquivos
    std::sort(images_filenames.begin(), images_filenames.end());
    std::sort(masks_filenames.begin(), masks_filenames.end());

    for (size_t i = 0; i < images_filenames.size() && i < masks_filenames.size(); ++i) {
        // Ler a imagem e a máscara
        cv::Mat img = cv::imread(images_filenames[i]);
        cv::Mat mask = cv::imread(masks_filenames[i], cv::IMREAD_GRAYSCALE); // Carregar máscara em escala de cinza

        if (!img.empty() && !mask.empty()) {
            images.push_back(img);
            masks.push_back(mask);
        }
    }
}

// Função para aplicar CLAHE
cv::Mat apply_clahe(const cv::Mat& image) {
    // Convert the image to grayscale before applying CLAHE
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(2, 2));
    cv::Mat cl1;
    clahe->apply(gray_image, cl1);

    return cl1;
}

// Função para aplicar blur (média) a uma imagem
cv::Mat apply_blur(const cv::Mat& image, const cv::Size& ksize = cv::Size(7, 7)) {
    cv::Mat blurred_image;
    cv::blur(image, blurred_image, ksize);
    return blurred_image;
}

// Função para aplicar binarização de Otsu a uma imagem
cv::Mat apply_otsu(const cv::Mat& image) {
    cv::Mat otsu;
    cv::threshold(image, otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    return otsu;
}

// Função para aplicar a detecção de bordas Canny a uma imagem
cv::Mat apply_canny(const cv::Mat& image, double threshold1 = 100, double threshold2 = 200) {
    cv::Mat edges;
    cv::Canny(image, edges, threshold1, threshold2);
    return edges;
}

// Função para aplicar o fechamento morfológico após a detecção de bordas com Canny
cv::Mat apply_closing(const cv::Mat& canny_image, const cv::Size& kernel_size = cv::Size(11, 11)) {
    // Criar um elemento estruturante
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    // Aplicar a operação de fechamento
    cv::Mat closed_image;
    cv::morphologyEx(canny_image, closed_image, cv::MORPH_CLOSE, kernel);
    return closed_image;
}

// Função para preencher áreas fechadas nas bordas detectadas por Canny
cv::Mat fill_closed_areas(const cv::Mat& canny_image) {
    // Encontrar contornos
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Criar uma imagem para desenhar os contornos preenchidos
    cv::Mat filled_image = cv::Mat::zeros(canny_image.size(), canny_image.type());

    // Preencher contornos
    cv::drawContours(filled_image, contours, -1, cv::Scalar(255), cv::FILLED);

    return filled_image;
}

// Função para aplicar erosão
cv::Mat apply_erosion(const cv::Mat& closed_image, const cv::Size& kernel_size = cv::Size(5, 5)) {
    // Criar um elemento estruturante elíptico
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    // Aplicar a operação de erosão
    cv::Mat eroded_image;
    cv::erode(closed_image, eroded_image, kernel, cv::Point(-1, -1), 2);
    return eroded_image;
}

// Função para aplicar dilatação
cv::Mat apply_dilation(const cv::Mat& eroded_image, const cv::Size& kernel_size = cv::Size(5, 5)) {
    // Criar um elemento estruturante elíptico
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    // Aplicar a operação de dilatação
    cv::Mat dilated_image;
    cv::dilate(eroded_image, dilated_image, kernel, cv::Point(-1, -1), 3);
    return dilated_image;
}

// Função para calcular a distância do centro de um contorno
double distance_from_center(const std::vector<cv::Point>& contour, const cv::Point& center) {
    cv::Moments M = cv::moments(contour);
    if (M.m00 != 0) {
        int cx = static_cast<int>(M.m10 / M.m00);
        int cy = static_cast<int>(M.m01 / M.m00);
        return std::sqrt((cx - center.x) * (cx - center.x) + (cy - center.y) * (cy - center.y));
    }
    return std::numeric_limits<double>::infinity();
}

// Função para manter as duas maiores regiões brancas mais próximas do centro
cv::Mat keep_largest_two_near_center(const cv::Mat& image) {
    // Encontrar contornos
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Calcular o centro da imagem
    cv::Point center(image.cols / 2, image.rows / 2);

    // Calcular a área e a distância do centro para cada contorno
    std::vector<std::tuple<std::vector<cv::Point>, double, double>> contours_info;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        double distance = distance_from_center(contour, center);
        contours_info.emplace_back(contour, area, distance);
    }

    // Ordenar os contornos pela distância do centro
    std::sort(contours_info.begin(), contours_info.end(), [](const auto& a, const auto& b) {
        return std::get<2>(a) < std::get<2>(b);
    });

    // Manter os 10 contornos mais próximos do centro para análise
    if (contours_info.size() > 10) {
        contours_info.resize(10);
    }

    // Ordenar os contornos mais próximos pela área
    std::sort(contours_info.begin(), contours_info.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b);
    });

    // Manter apenas os dois maiores contornos entre os mais próximos
    std::vector<std::vector<cv::Point>> largest_two_contours;
    for (size_t i = 0; i < std::min(contours_info.size(), size_t(2)); ++i) {
        largest_two_contours.push_back(std::get<0>(contours_info[i]));
    }

    // Criar uma máscara para manter apenas essas duas regiões
    cv::Mat mask = cv::Mat::zeros(image.size(), image.type());
    cv::drawContours(mask, largest_two_contours, -1, cv::Scalar(255), cv::FILLED);

    // Aplicar a máscara à imagem original para manter apenas as duas maiores regiões brancas
    cv::Mat result_image;
    cv::bitwise_and(image, mask, result_image);

    return result_image;
}

// Função do algoritmo "Rolling-Ball"
std::pair<cv::Mat, cv::Mat> rolling_ball(const cv::Mat& mask, int radius) {
    // Criação do elemento estruturante esférico
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius, radius));

    // Aplicação da operação de fechamento
    cv::Mat closed;
    cv::morphologyEx(mask, closed, cv::MORPH_CLOSE, kernel);

    // Aplicação da operação de abertura
    cv::Mat opened;
    cv::morphologyEx(mask, opened, cv::MORPH_OPEN, kernel);

    return {closed, opened};
}

// Função para exibir imagens com Canny aplicado e com áreas fechadas preenchidas
void display_canny_and_filled(const std::vector<cv::Mat>& canny_images, const std::vector<cv::Mat>& filled_images, int index) {
    // Exibir a imagem com Canny aplicado
    cv::imshow("Imagem com Canny", canny_images[index]);

    // Exibir a imagem com áreas fechadas preenchidas
    cv::imshow("Imagem com Áreas Fechadas Preenchidas", filled_images[index]);

    // Aguardar o usuário pressionar uma tecla
    cv::waitKey(0);

    // Fechar as janelas
    cv::destroyAllWindows();
}

// Função para exibir imagens com preenchimento aplicado e com erosão aplicada
void display_closed_and_eroded(const std::vector<cv::Mat>& filled_images, const std::vector<cv::Mat>& eroded_images, int index) {
    // Exibir a imagem com preenchimento aplicado
    cv::imshow("Imagem com Preenchimento", filled_images[index]);

    // Exibir a imagem com erosão aplicada
    cv::imshow("Imagem com Erosão", eroded_images[index]);

    // Aguardar o usuário pressionar uma tecla
    cv::waitKey(0);

    // Fechar as janelas
    cv::destroyAllWindows();
}

// Função para exibir imagens com erosão aplicada e com dilatação aplicada
void display_eroded_and_dilated(const std::vector<cv::Mat>& eroded_images, const std::vector<cv::Mat>& dilated_images, int index) {
    // Exibir a imagem com erosão aplicada
    cv::imshow("Imagem com Erosão", eroded_images[index]);

    // Exibir a imagem com dilatação aplicada
    cv::imshow("Imagem com Dilatação", dilated_images[index]);

    // Aguardar o usuário pressionar uma tecla
    cv::waitKey(0);

    // Fechar as janelas
    cv::destroyAllWindows();
}

// Função para exibir imagens com filtro aplicado e com algoritmo Rolling-Ball aplicado
void display_filtered_and_rolling_ball(const std::vector<cv::Mat>& filtered_images, const std::vector<cv::Mat>& rolling_ball_images, int index) {
    // Exibir a imagem com filtro aplicado
    cv::imshow("Imagem com Filtro", filtered_images[index]);

    // Exibir a imagem com algoritmo Rolling-Ball aplicado
    cv::imshow("Imagem com Rolling-Ball", rolling_ball_images[index]);

    // Aguardar o usuário pressionar uma tecla
    cv::waitKey(0);

    // Fechar as janelas
    cv::destroyAllWindows();
}

// Função para calcular a precisão
double calculate_precision(const cv::Mat& true_mask, const cv::Mat& pred_mask) {
    int tp = cv::countNonZero(true_mask & pred_mask); // True Positives
    int fp = cv::countNonZero(~true_mask & pred_mask); // False Positives
    if (tp + fp == 0) return 0; // Evitar divisão por zero
    return tp / static_cast<double>(tp + fp);
}

// Função para calcular o recall
double calculate_recall(const cv::Mat& true_mask, const cv::Mat& pred_mask) {
    int tp = cv::countNonZero(true_mask & pred_mask); // True Positives
    int fn = cv::countNonZero(true_mask & ~pred_mask); // False Negatives
    if (tp + fn == 0) return 0; // Evitar divisão por zero
    return tp / static_cast<double>(tp + fn);
}

// Função para calcular o F1 score
double calculate_f1(double precision, double recall) {
    if (precision + recall == 0) return 0; // Evitar divisão por zero
    return 2 * (precision * recall) / (precision + recall);
}

// Função para calcular o IoU
double calculate_iou(const cv::Mat& true_mask, const cv::Mat& pred_mask) {
    int intersection = cv::countNonZero(true_mask & pred_mask);
    int union_area = cv::countNonZero(true_mask | pred_mask);
    if (union_area == 0) return 0; // Evitar divisão por zero
    return intersection / static_cast<double>(union_area);
}

// Avaliação da segmentação
std::tuple<double, double, double, double> evaluate_segmentation(const std::vector<cv::Mat>& true_masks, const std::vector<cv::Mat>& predicted_masks) {
    std::vector<double> precisions;
    std::vector<double> recalls;
    std::vector<double> f1_scores;
    std::vector<double> ious;

    for (size_t i = 0; i < true_masks.size(); ++i) {
        cv::Mat true_mask = true_masks[i];
        cv::Mat pred_mask = predicted_masks[i];

        // Garantir que as máscaras tenham o mesmo tamanho
        if (true_mask.size() != pred_mask.size()) {
            cv::resize(pred_mask, pred_mask, true_mask.size());
        }

        // Binarizar as máscaras
        cv::threshold(true_mask, true_mask, 0.5, 1, cv::THRESH_BINARY);
        cv::threshold(pred_mask, pred_mask, 0.5, 1, cv::THRESH_BINARY);

        // Calcular métricas
        double precision = calculate_precision(true_mask, pred_mask);
        double recall = calculate_recall(true_mask, pred_mask);
        double f1 = calculate_f1(precision, recall);
        double iou = calculate_iou(true_mask, pred_mask);

        precisions.push_back(precision);
        recalls.push_back(recall);
        f1_scores.push_back(f1);
        ious.push_back(iou);
    }

    // Calcular métricas médias
    double avg_precision = std::accumulate(precisions.begin(), precisions.end(), 0.0) / precisions.size();
    double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
    double avg_f1 = std::accumulate(f1_scores.begin(), f1_scores.end(), 0.0) / f1_scores.size();
    double avg_iou = std::accumulate(ious.begin(), ious.end(), 0.0) / ious.size();

    return {avg_precision, avg_recall, avg_f1, avg_iou};
}

int main() {
    // Definir os caminhos para as pastas de imagens e máscaras
    std::string path_to_images = "/Users/celsosoares/Desktop/Shenzhen_datase/img";
    std::string path_to_masks = "/Users/celsosoares/Desktop/Shenzhen_datase/mask";

    // Vetores para armazenar imagens e máscaras
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> masks;

    // Carregar imagens e máscaras
    load_images_and_masks(path_to_images, path_to_masks, images, masks);

    // Aplicar CLAHE a todas as imagens
    std::vector<cv::Mat> clahe_images;
    for (const auto& image : images) {
        clahe_images.push_back(apply_clahe(image));
    }

    // Exibir a imagem após a aplicação de CLAHE
    cv::imshow("Imagem com CLAHE", clahe_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Aplicar blur a todas as imagens com CLAHE aplicado
    std::vector<cv::Mat> blurred_images;
    for (const auto& image : clahe_images) {
        blurred_images.push_back(apply_blur(image));
    }

    // Exibir a imagem após a aplicação de blur
    cv::imshow("Imagem com Blur", blurred_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Aplicar Otsu a todas as imagens com blur
    std::vector<cv::Mat> otsu_images;
    for (const auto& image : blurred_images) {
        otsu_images.push_back(apply_otsu(image));
    }

    // Exibir a imagem após a aplicação de Otsu
    cv::imshow("Imagem com Otsu", otsu_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Aplicar Canny a todas as imagens com Otsu
    std::vector<cv::Mat> canny_images;
    for (const auto& image : otsu_images) {
        canny_images.push_back(apply_canny(image));
    }

    // Exibir a imagem após a aplicação de Canny
    cv::imshow("Imagem com Canny", canny_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Aplicar fechamento a todas as imagens com Canny
    std::vector<cv::Mat> closed_images;
    for (const auto& image : canny_images) {
        closed_images.push_back(apply_closing(image));
    }

    // Exibir a imagem após a aplicação de fechamento
    cv::imshow("Imagem com Fechamento", closed_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Aplicar preenchimento de áreas fechadas a todas as imagens com fechamento
    std::vector<cv::Mat> filled_images;
    for (const auto& image : closed_images) {
        filled_images.push_back(fill_closed_areas(image));
    }

    // Exibir a imagem após o preenchimento de áreas fechadas
    cv::imshow("Imagem com Áreas Fechadas Preenchidas", filled_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Aplicar fechamento e depois erosão a todas as imagens com preenchimento
    std::vector<cv::Mat> closed_and_eroded_images;
    for (const auto& image : filled_images) {
        closed_and_eroded_images.push_back(apply_erosion(image));
    }

    // Exibir a imagem após a aplicação de erosão
    cv::imshow("Imagem com Erosão", closed_and_eroded_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Aplicar dilatação a todas as imagens com erosão
    std::vector<cv::Mat> dilated_images;
    for (const auto& image : closed_and_eroded_images) {
        dilated_images.push_back(apply_dilation(image));
    }

    // Exibir a imagem após a aplicação de dilatação
    cv::imshow("Imagem com Dilatação", dilated_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Aplicar a função para manter as duas maiores regiões brancas mais próximas do centro em todas as imagens com Otsu
    std::vector<cv::Mat> filtered_images;
    for (const auto& image : dilated_images) {
        filtered_images.push_back(keep_largest_two_near_center(image));
    }

    // Exibir a imagem após manter as duas maiores regiões brancas mais próximas do centro
    cv::imshow("Imagem com Regiões Brancas mais Próximas do Centro", filtered_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Aplicar o algoritmo "Rolling-Ball" nas imagens filtradas
    int radius_internal = 25;
    int radius_external = 23;

    std::vector<cv::Mat> rolling_ball_images;
    for (const auto& image : filtered_images) {
        auto [closed_internal, opened_internal] = rolling_ball(image, radius_internal);
        auto [closed_external, opened_external] = rolling_ball(opened_internal, radius_external);
        rolling_ball_images.push_back(closed_external);
    }

    // Exibir imagem com filtro aplicado e com algoritmo Rolling-Ball aplicado
    cv::imshow("Imagem com Rolling-Ball", rolling_ball_images[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Avaliar a segmentação
    auto [avg_precision, avg_recall, avg_f1, avg_iou] = evaluate_segmentation(masks, rolling_ball_images);

    // Exibir as métricas
    std::cout << "Precision: " << avg_precision << std::endl;
    std::cout << "Recall: " << avg_recall << std::endl;
    std::cout << "F1 Score: " << avg_f1 << std::endl;
    std::cout << "IoU: " << avg_iou << std::endl;

    return 0;
}

