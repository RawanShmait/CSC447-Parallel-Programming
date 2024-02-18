#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define IMAGE_WIDTH 800
#define IMAGE_HEIGHT 800
#define MAX_ITERATIONS 1000

int computeMB(double x, double y)
{
    double r = 0.0, i = 0.0;
    int iterations = 0;
    while (r * r + i * i <= 4.0 && iterations < MAX_ITERATIONS)
    {
        double temp = r * r - i * i + x;
        i = 2.0 * r * i + y;
        r = temp;
        iterations++;
    }
    return iterations;
}

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    clock_t start_time = clock();

    int rowsPerProcess = IMAGE_HEIGHT / size;
    int start_row = rank * rowsPerProcess;
    int end_row = (rank + 1) * rowsPerProcess;

    if (rank == size - 1)
        end_row = IMAGE_HEIGHT;

    int *image = (int *)malloc(IMAGE_WIDTH * (end_row - start_row) * sizeof(int));

    for (int j = start_row; j < end_row; j++)
    {
        for (int k = 0; k < IMAGE_WIDTH; k++)
        {
            double x = (k - IMAGE_WIDTH / 2.0) * 4.0 / IMAGE_WIDTH;
            double y = (j - IMAGE_HEIGHT / 2.0) * 4.0 / IMAGE_HEIGHT;
            image[(j - start_row) * IMAGE_WIDTH + k] = computeMB(x, y);
        }
    }

    if (rank == 0)
    {
        int *result = (int *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(int));
        MPI_Gather(image, IMAGE_WIDTH * rowsPerProcess, MPI_INT, result, IMAGE_WIDTH * rowsPerProcess, MPI_INT, 0, MPI_COMM_WORLD);

        FILE *fp = fopen("mandelbrot.ppm", "wb");
        fprintf(fp, "P6\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
        for (int j = 0; j < IMAGE_HEIGHT; j++)
        {
            for (int k = 0; k < IMAGE_WIDTH; k++)
            {
                int index = j * IMAGE_WIDTH + k;
                unsigned char r, g, b;
                if (result[index] == MAX_ITERATIONS)
                    r = g = b = 0;
                else
                {
                    r = (result[index] % 256);
                    g = (result[index] % 256);
                    b = (result[index] % 256);
                }
                fwrite(&r, 1, 1, fp);
                fwrite(&g, 1, 1, fp);
                fwrite(&b, 1, 1, fp);
            }
        }
        fclose(fp);
        free(result);
    }

    else
        MPI_Gather(image, IMAGE_WIDTH * rowsPerProcess, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

    free(image);
    MPI_Finalize();

    if (rank == 0)
    {
        clock_t end_time = clock();
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("(Static) Time taken: %.4f seconds\n", elapsed_time);
    }

    return 0;
}
