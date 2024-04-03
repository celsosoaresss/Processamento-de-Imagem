// Função percorre toda a matriz da imagem (imagem em escala de cinza) e divide os valores dos pixels por 2.
// Certifique-se de alterar o caminho da imagem na linha 26.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void quantizeImage(Mat& src, Mat& dst, int totalLevels) {
    
    int interval = 256 / totalLevels; // Calcula o intervalo de quantização
    src.copyTo(dst); // cópia da imagem

    for (int y = 0; y < src.rows; y++) { 
        for (int x = 0; x < src.cols; x++) {
            uchar& pixel = dst.at<uchar>(y, x);
            pixel = (pixel / interval) * interval + interval / 2; //substituir os valores da matriz
        }
    }
}

int main() {
    
    Mat src = imread("C:/Users/admin/Desktop/PI/src/imagens/flores_laranjas.jpeg", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "erro ao abrir a imagem" << endl;
        return -1;
    }

    double escalaImagem = 0.25;  //redimensionamento da imagem

    for (int totalLevels = 256; totalLevels >= 2; totalLevels /= 2) { //quantidade de níveis inicial/2 a cada interação
        Mat quantizedImage;
        quantizeImage(src, quantizedImage, totalLevels);

        // resize da imagem
        Mat redimencionarrImagem;
        resize(quantizedImage, redimencionarrImagem, Size(), escalaImagem, escalaImagem, INTER_LINEAR);

        //nome da janela
        string nomeJanela = "Quantizada - Níveis: " + to_string(totalLevels);
        imshow(nomeJanela, redimencionarrImagem);
        waitKey(0); 
    }

    return 0;
}
