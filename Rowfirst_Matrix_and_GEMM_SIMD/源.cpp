#include"MyMatrix.h"

using namespace std;

//#define M 1//C_out=3
//#define N 1024*1024//H_out*W_out=8*8
//#define K 9//C_in*H_k*W_k=3*3*3
//

//
#define M 32//C_out=3
#define N 32//H_out*W_out=8*8
#define K 32//C_in*H_k*W_k=3*3*3

//#define M 1024//C_out=3
//#define N 1024//H_out*W_out=8*8
//#define K 1024//C_in*H_k*W_k=3*3*3


int main() {

	

	Matrix A(M,K,0);
	Matrix B(K,N,0);
	Matrix C(A.rows(),B.cols(),0);


	for (int i = 0; i < A.rows(); i++) {
		for (int j = 0; j < A.cols(); j++) {
			A(i, j) = i*A.cols() + j + 1;
		}
	}
	for (int i = 0; i < B.rows(); i++) {
		for (int j = 0; j < B.cols(); j++) {
			if (i == j||i==0||j==0) {
				B(i, j) = 1;
			}
		}
	}




	//C = Matrix::multi(A, B);
	//C = Matrix::multi_midk(A, B);
	//C = Matrix::multi_register(A, B);
	//C = Matrix::multi_loopunrolling(A, B);
	//C =Matrix::multi_avx(A, B);
	//C = Matrix::multi_avx_blk(A, B);
	//C = Matrix::multi_avx_unrollx2(A, B);
	//C = Matrix::multi_avx_unrollx4(A, B);
	//C = Matrix::multi_avx_unrollx4_blk(A, B);
	C = Matrix::multi_avx_unrollx4_blk_omp(A, B);
	//C = Matrix::multi_avx_unrollx8_ijk(A, B);
	//C = Matrix::multi_avx_unrollx8_jik(A, B);
	//C = Matrix::multi_avx_unrollx8_omp(A, B);



	cout << "A: " << endl;A.print();cout << "B: " << endl;B.print();cout << "C: " << endl;C.print();


	return 0;
}


