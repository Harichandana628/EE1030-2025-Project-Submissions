#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMAGE_SIZE 256

// Safe memory allocation
void* my_malloc(int size) {
    void *p = malloc(size);
    if (p == NULL) {
        printf("Error in allocate memory!\n");
        exit(1);
    }
    return p;
}

// Read PGM image
int read_pgm(char *filename, unsigned char **image, int *width, int *height) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Cannot open file %s\n", filename);
        return 0;
    }
    
    char magic[3];
    fscanf(file, "%s", magic);
    if (strcmp(magic, "P2") != 0) {
        printf("Error: Only P2 format supported!\n");
        fclose(file);
        return 0;
    }
    
    fscanf(file, "%d %d", width, height);
    
    int max_val;
    fscanf(file, "%d", &max_val);
    
    if (*width != IMAGE_SIZE || *height != IMAGE_SIZE) {
        printf("Error: Image must be 256x256!\n");
        fclose(file);
        return 0;
    }
    
    *image = (unsigned char*)my_malloc(IMAGE_SIZE * IMAGE_SIZE);
    
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        int pixel;
        fscanf(file, "%d", &pixel);
        (*image)[i] = (unsigned char)pixel;
    }
    
    fclose(file);
    return 1;
}

// Write PGM image
int write_pgm(char *filename, unsigned char *image, int width, int height) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Cannot create file %s\n", filename);
        return 0;
    }
    
    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(file, "%d", image[i * width + j]);
            if (j < width - 1) fprintf(file, " ");
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    return 1;
}

// Matrix transpose operation
void transpose_matrix(double *A, double *AT) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            AT[j * IMAGE_SIZE + i] = A[i * IMAGE_SIZE + j];
        }
    }
}

// Matrix multiplication operation
void multiply_matrices(double *A, double *B, double *C) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < IMAGE_SIZE; k++) {
                sum += A[i * IMAGE_SIZE + k] * B[k * IMAGE_SIZE + j];
            }
            C[i * IMAGE_SIZE + j] = sum;
        }
    }
}

// Eigenvalue decomposition computation
void jacobi_method(double *S, double *V, double *eigenvalues) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            if (i == j) {
                V[i * IMAGE_SIZE + j] = 1.0;
            } else {
                V[i * IMAGE_SIZE + j] = 0.0;
            }
        }
    }
    
    for (int iteration = 0; iteration < 50; iteration++) {
        for (int p = 0; p < IMAGE_SIZE; p++) {
            for (int q = p + 1; q < IMAGE_SIZE; q++) {
                double app = S[p * IMAGE_SIZE + p];
                double aqq = S[q * IMAGE_SIZE + q];
                double apq = S[p * IMAGE_SIZE + q];
                
                if (fabs(apq) < 0.0001) continue;
                
                double angle = 0.5 * atan2(2 * apq, aqq - app);
                double c = cos(angle);
                double s = sin(angle);
                
                for (int k = 0; k < IMAGE_SIZE; k++) {
                    double temp1 = S[p * IMAGE_SIZE + k];
                    double temp2 = S[q * IMAGE_SIZE + k];
                    S[p * IMAGE_SIZE + k] = c * temp1 - s * temp2;
                    S[q * IMAGE_SIZE + k] = s * temp1 + c * temp2;
                }
                
                for (int k = 0; k < IMAGE_SIZE; k++) {
                    double temp1 = S[k * IMAGE_SIZE + p];
                    double temp2 = S[k * IMAGE_SIZE + q];
                    S[k * IMAGE_SIZE + p] = c * temp1 - s * temp2;
                    S[k * IMAGE_SIZE + q] = s * temp1 + c * temp2;
                }
                
                for (int k = 0; k < IMAGE_SIZE; k++) {
                    double temp1 = V[k * IMAGE_SIZE + p];
                    double temp2 = V[k * IMAGE_SIZE + q];
                    V[k * IMAGE_SIZE + p] = c * temp1 - s * temp2;
                    V[k * IMAGE_SIZE + q] = s * temp1 + c * temp2;
                }
            }
        }
    }
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        eigenvalues[i] = S[i * IMAGE_SIZE + i];
    }
}

// Sort eigenvalues descending
void sort_eigenvalues(double *eigenvalues, double *V) {
    for (int i = 0; i < IMAGE_SIZE - 1; i++) {
        for (int j = 0; j < IMAGE_SIZE - i - 1; j++) {
            if (eigenvalues[j] < eigenvalues[j + 1]) {
                double temp_eval = eigenvalues[j];
                eigenvalues[j] = eigenvalues[j + 1];
                eigenvalues[j + 1] = temp_eval;
                
                for (int k = 0; k < IMAGE_SIZE; k++) {
                    double temp_vec = V[k * IMAGE_SIZE + j];
                    V[k * IMAGE_SIZE + j] = V[k * IMAGE_SIZE + j + 1];
                    V[k * IMAGE_SIZE + j + 1] = temp_vec;
                }
            }
        }
    }
}

// Compute SVD decomposition
void simple_svd(double *A, double *U, double *S, double *V) {
    double *AT = (double*)my_malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    double *ATA = (double*)my_malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    
    transpose_matrix(A, AT);
    multiply_matrices(AT, A, ATA);
    
    double *eigenvalues = (double*)my_malloc(IMAGE_SIZE * sizeof(double));
    double *temp_matrix = (double*)my_malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    
    memcpy(temp_matrix, ATA, IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    jacobi_method(temp_matrix, V, eigenvalues);
    
    sort_eigenvalues(eigenvalues, V);
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        if (eigenvalues[i] > 0) {
            S[i] = sqrt(eigenvalues[i]);
        } else {
            S[i] = 0.0;
        }
    }
    
    double *AV = (double*)my_malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    multiply_matrices(A, V, AV);
    
    for (int j = 0; j < IMAGE_SIZE; j++) {
        if (S[j] > 0.0001) {
            for (int i = 0; i < IMAGE_SIZE; i++) {
                U[i * IMAGE_SIZE + j] = AV[i * IMAGE_SIZE + j] / S[j];
            }
        } else {
            for (int i = 0; i < IMAGE_SIZE; i++) {
                U[i * IMAGE_SIZE + j] = 0.0;
            }
        }
    }
    
    free(AT);
    free(ATA);
    free(eigenvalues);
    free(temp_matrix);
    free(AV);
}

// Reconstruct image approximation
void reconstruct_image(double *U, double *S, double *V, int k, double *result) {
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        result[i] = 0.0;
    }
    
    for (int comp = 0; comp < k; comp++) {
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                result[i * IMAGE_SIZE + j] += U[i * IMAGE_SIZE + comp] * S[comp] * V[j * IMAGE_SIZE + comp];
            }
        }
    }
}

// Calculate reconstruction error
double calculate_error(double *original, double *reconstructed) {
    double error = 0.0;
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        double diff = original[i] - reconstructed[i];
        error += diff * diff;
    }
    return sqrt(error);
}

// Main program entry
int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Error\n");
        return 1;
    }
    
    char *input_file = argv[1];
    char *output_file = argv[2];
    int k = atoi(argv[3]);
    
    if (k < 1 || k > 100) {
        printf("k must be between 1 and 100\n");
        return 1;
    }
    
    printf("Reading image %s...\n", input_file);
    
    unsigned char *image_data = NULL;
    int width, height;
    
    if (!read_pgm(input_file, &image_data, &width, &height)) {
        return 1;
    }
    
    printf("Image loaded: %dx%d pixels\n", width, height);
    
    double *A = (double*)my_malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        A[i] = (double)image_data[i];
    }
    
    double *U = (double*)my_malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    double *S = (double*)my_malloc(IMAGE_SIZE * sizeof(double));
    double *V = (double*)my_malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    
    simple_svd(A, U, S, V);
    
    double *reconstructed = (double*)my_malloc(IMAGE_SIZE * IMAGE_SIZE * sizeof(double));
    reconstruct_image(U, S, V, k, reconstructed);
    
    double error = calculate_error(A, reconstructed);
    printf("Reconstruction error: %.6f\n", error);
    
    unsigned char *output_data = (unsigned char*)my_malloc(IMAGE_SIZE * IMAGE_SIZE);
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        int pixel_value = (int)(reconstructed[i] + 0.5); // Round to nearest integer
        if (pixel_value < 0) pixel_value = 0;
        if (pixel_value > 255) pixel_value = 255;
        output_data[i] = (unsigned char)pixel_value;
    }
    
  
    if (!write_pgm(output_file, output_data, IMAGE_SIZE, IMAGE_SIZE)) {
        return 1;
    }
    
    printf("Done! \n");
    
    free(image_data);
    free(A);
    free(U);
    free(S);
    free(V);
    free(reconstructed);
    free(output_data);
    
    return 0;
}
