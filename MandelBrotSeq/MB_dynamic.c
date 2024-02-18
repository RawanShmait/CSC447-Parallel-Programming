#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

typedef struct
{
    float real;
    float imag;
} complex;

typedef struct
{
    int x;
    int y;
    int c;
} mb;

typedef enum
{
    data_tag,
    result_tag,
    terminator_tag
} tag;

float real_min = -2.0;
float real_max = 1.0;
float imag_min = -1.5;
float imag_max = 1.5;

#define disp_width 400
#define disp_height 400
#define block_width 10

int arr[disp_height][disp_width];

int p;
MPI_Datatype MPI_Mand;

typedef unsigned char U8;

void save(const char *file_name, int width, int height)
{
    FILE *f = fopen(file_name, "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            unsigned char r, g, b;
            r = g = b = (unsigned char)arr[i][j];
            fwrite(&r, 1, 1, f);
            fwrite(&g, 1, 1, f);
            fwrite(&b, 1, 1, f);
        }
    }

    fclose(f);
}

int cal_pixel(complex c)
{
    int count, max;
    complex z;
    float temp, lengthsq;
    max = 255;
    z.real = 0;
    z.imag = 0;
    count = 0; /* number of iterations */
    do
    {
        temp = z.real * z.real - z.imag * z.imag + c.real;
        z.imag = 2 * z.real * z.imag + c.imag;
        z.real = temp;
        lengthsq = z.real * z.real + z.imag * z.imag;
        count++;
    } while ((lengthsq < 4.0) && (count < max));
    return count - 1;
}

void display(mb data)
{
    arr[data.y][data.x] = data.c;
}

void calc_block_and_send(int col, int rank)
{
    complex c;
    mb m[block_width * disp_height];
    int cnt = 0;
    float scale_real = (real_max - real_min) / disp_width;
    float scale_imag = (imag_max - imag_min) / disp_height;

    for (int x = col; x < col + block_width; x++)
    {
        for (int y = 0; y < disp_height; y++)
        {
            c.real = real_min + ((float)x * scale_real);
            c.imag = imag_min + ((float)y * scale_imag);
            int color = cal_pixel(c);
            m[cnt].x = x;
            m[cnt].y = y;
            m[cnt].c = color;
            cnt++;
        }
    }

    MPI_Send(&m, cnt, MPI_Mand, 0, result_tag, MPI_COMM_WORLD);
}

void master()
{
    int count = 0, col = 0, slave_rank;
    MPI_Status status;
    mb data[block_width * disp_height];

    for (int i = 1; i < p; i++, col += block_width, count++)
        MPI_Send(&col, 1, MPI_INT, i, data_tag, MPI_COMM_WORLD);

    do
    {
        MPI_Recv(&data, block_width * disp_height, MPI_Mand, MPI_ANY_SOURCE,
                 result_tag, MPI_COMM_WORLD, &status);
        for (int i = 0; i < block_width * disp_height; i++)
            display(data[i]);
        count--;
        slave_rank = status.MPI_SOURCE;
        if (col < disp_width)
        {
            MPI_Send(&col, 1, MPI_INT, slave_rank, data_tag, MPI_COMM_WORLD);
            col += block_width;
            count++;
        }
        else
            MPI_Send(&col, 1, MPI_INT, slave_rank, terminator_tag, MPI_COMM_WORLD);
    } while (count > 0);
}

void slave(int rank)
{
    MPI_Status status;
    int col;
    MPI_Recv(&col, 1, MPI_INT, 0, data_tag, MPI_COMM_WORLD, &status);

    while (status.MPI_TAG == data_tag)
    {
        calc_block_and_send(col, rank);
        MPI_Recv(&col, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
}

int main(int argc, char *argv[])
{
    double start, end, duration, global;
    int me;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    MPI_Type_contiguous(3, MPI_INT, &MPI_Mand);
    MPI_Type_commit(&MPI_Mand);

    MPI_Barrier(MPI_COMM_WORLD);
    clock_t start_time = clock();

    if (me == 0)
        master();
    else
        slave(me);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&duration, &global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (me == 0)
    {
        clock_t end_time = clock();
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("(Dynamic) Time taken: %.4f seconds\n", elapsed_time);
        printf("Global runtime is %f\n", global);
        char filename[256];
        sprintf(filename, "dynamic-mandelbrot-output.ppm");
        save(filename, disp_width, disp_height);
        printf("[+] save %s\n", filename);
    }

    MPI_Finalize();
    return 0;
}
