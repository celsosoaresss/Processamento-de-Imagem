//**** para rodar os algorítimos, basta deixar o primeiro ou o segundo comentado*****




// ***** PRIMEIRO ALGORITMO ***** //
// Função percorre toda a matriz da imagem (imagem em escala de cinza) e divide os valores dos pixels por 2.

/*
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
}*/


// ***** SEGUNDO ALGORITMO ***** //

// uma segunda abordagem é substituindo o laço for por um while
// valor de total de nível de quantização é cravado no variável "totalLevels"
// o valor mínimo de quantização segue sendo >=2
// o valor da imagem quantizado seguem sendo divididos por 

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string> // Para usar to_string

using namespace cv;
using namespace std;

void quantizeImage(Mat& src, Mat& dst, int totalLevels) {
    if (totalLevels < 2) return; // Garante que haja pelo menos dois níveis

    int interval = 256 / totalLevels;
    src.copyTo(dst);

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            uchar& pixel = dst.at<uchar>(y, x);
            pixel = (pixel / interval) * interval + interval / 2;
        }
    }
}

int main() {
    Mat src = imread("C:/Users/admin/Desktop/PI/src/imagens/flores_laranjas.jpeg", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Erro ao abrir a imagem!" << endl;
        return -1;
    }

    double scaleFactor = 0.25; 

    int totalLevels = 32; // Inicia com 8 níveis de quantização
    while (totalLevels >= 2) {
        Mat quantizedImage;
        quantizeImage(src, quantizedImage, totalLevels);

        Mat resizedImage;
        resize(quantizedImage, resizedImage, Size(), scaleFactor, scaleFactor, INTER_LINEAR);

        string windowName = "Quantizada - Níveis: " + to_string(totalLevels);
        imshow(windowName, resizedImage);
        waitKey(0); 
        totalLevels /= 2; 
    }

    return 0;
}
