#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../Utils/utils.h"

int rank;
int n_processes;
int width;
int height;
int iterations;

// Vectorize a single channel for sending
unsigned char *pack_channel(Channels *vec, int length, int channel_id)
{
    unsigned char *channel = malloc(length * sizeof(unsigned char));
    for (int i = 0; i < length; i++)
        channel[i] = vec[i].channel[channel_id];

    return channel;
}

// Unpacks the given channels into the first 3 channels of the array, used for RGB representation
Channels *unpack_rgb_channels(unsigned char *red, unsigned char *green, unsigned char *blue, int length)
{
    Channels *colors = malloc(length * sizeof(Channels));
    for (int i = 0; i < length; i++)
    {
        colors[i].channel[0] = red[i];
        colors[i].channel[1] = green[i];
        colors[i].channel[2] = blue[i];
    }

    return colors;
}

// Standardizes a batch 1 image into the range 0-255
Channels **normalize_batch(Channels **img, int num_channels, int top, int start, int end, int offset)
{
    float upscale_factor = 255.f / top;
    int size = end - start;
    for (int i = 1 + offset; i < size + 2 * iterations - offset - 1; i++)
        for (int j = 1; j < width + 1; j++)
            for (int c = 0; c < num_channels; c++)
                img[i][j].channel[c] = clamp_to_byte(upscale_factor * img[i][j].channel[c]);

    return img;
}

// Applies the vertical part of the spatial sepratable convolution
void conv_vertical(Channels **img, int num_channels, int start, int end, int offset)
{
    int size = end - start;
    float K[channel_count] = {1.f / channel_count, 2.f / channel_count, 1.f / channel_count};

    // The array is extended 'iterations' amount of rows in each direction
    // And bordered with 0 on width
    Channels **bordered_img = new_channel_array(size + 2 * iterations, width + 2);

    for (int i = 0; i < size + 2 * iterations; i++)
        for (int j = 0; j < width; j++)
            bordered_img[i][j + 1] = img[i][j];

    for (int i = 1 + offset; i < size + 2 * iterations - offset - 1; i++)
        for (int j = 1; j < width + 1; j++)
        {
            float *final_pixel = calloc(num_channels, sizeof(float));
            for (int m = -1, k = 0; m <= 1; m++, k++)
                // In case of margins, those can't be extended, so we have to treat them as bordered with 0
                if (!((start == 0 && i < iterations) || (end == height && i > size + iterations - 1)))
                    for (int c = 0; c < num_channels; c++)
                        final_pixel[c] += bordered_img[i + m][j].channel[c] * K[k];

            for (int c = 0; c < num_channels; c++)
                img[i][j - 1].channel[c] = clamp_to_byte(final_pixel[c]);

            free(final_pixel);
        }
}

// Applies the horizonal part of the spatial sepratable convolution
int conv_horizontal(Channels **img, int num_channels, int start, int end, int offset)
{
    int size = end - start;
    int top = 0;
    float K[channel_count] = {-1.f / channel_count, 0 / channel_count, 1.f / channel_count};

    Channels **bordered_img = new_channel_array(size + 2 * iterations, width + 2);

    for (int i = 0; i < size + 2 * iterations; i++)
        for (int j = 0; j < width; j++)
            bordered_img[i][j + 1] = img[i][j];

    for (int i = 1 + offset; i < size + 2 * iterations - offset - 1; i++)
        for (int j = 1; j < width + 1; j++)
        {
            float *final_pixel = calloc(num_channels, sizeof(float));
            for (int n = -1, k = 0; n <= 1; n++, k++)
                if (!((start == 0 && i < iterations) || (end == height && i > size + iterations - 1)))
                    for (int c = 0; c < num_channels; c++)
                        final_pixel[c] += bordered_img[i][j + n].channel[c] * K[k];

            for (int c = 0; c < num_channels; c++)
            {
                img[i][j - 1].channel[c] = clamp_to_byte(final_pixel[c]);
                // Also keep in mind the top range of the distribution here to avoid another traversal
                top = fmax(top, img[i][j - 1].channel[c]);
            }

            free(final_pixel);
        }

    return top;
}

/* Applies a depthwise convolution with a random kernel, increasing the number of channels
to the specified amount. Must be a multiple of the original arrays number of channels! */
void conv_depthwise_encode(Channels **img, int num_channels, int start, int end, int offset)
{

    int size = end - start;
    // This kernel should be learned by our network, but here we will randomly generate it
    float *K = get_kernel(offset);

    // Pool the channels with a stride of 'channel_count'
    for (int channel_id = channel_count; channel_id < num_channels; channel_id++)
    {
        for (int i = 1 + offset; i < size - offset - 1; i++)
            for (int j = 1; j < width + 1; j++)
            {
                float final_pixel = 0;
                for (int c = 0; c < channel_count; c++)
                    final_pixel += img[i][j].channel[channel_id % channel_count] * K[c];

                img[i][j].channel[channel_id] = clamp_to_byte(final_pixel);
            }
    }
    free(K);
}

/* Applies the depthwise convolution with a static kernel, transforming the array back
into a 3-channel image. */
void conv_depthwise_decode(Channels **img, int num_channels, int start, int end, int offset)
{
    int size = end - start;
    int multiplier = num_channels / channel_count;

    for (int i = 1 + offset; i < size - offset - 1; i++)
        for (int j = 1; j < width + 1; j++)
        {
            float red = 0;
            float green = 0;
            float blue = 0;

            // This has the effect of averaging out all the dimensions into 3 colors
            // An additional kernel could be used for more complex outputs
            for (int c = 0; c < num_channels; c += channel_count)
            {
                red += img[i][j].channel[c + 0];
                green += img[i][j].channel[c + 1];
                blue += img[i][j].channel[c + 2];
            }
            img[i][j].channel[0] = clamp_to_byte(red / multiplier);
            img[i][j].channel[1] = clamp_to_byte(green / multiplier);
            img[i][j].channel[2] = clamp_to_byte(blue / multiplier);
        }
}

/* Applies the depthwise separable convolution to the given image.
    - offset: Tells us how deep to convolve the extended rows of the bordered array
when applying multiple iterations at once. This helps reduce the number of transfers between
processes to 1, agnostic of the number of iterations.
    - channel_multiplier: Applies a polling step the the array, extending the number of channels
by the given amount.
*/
void conv_separable(Channels **img, int channel_multiplier, int start, int end, int offset)
{
    // First we apply the vertical kernel
    conv_vertical(img, channel_count, start, end, offset);
    // The applying the horizonal part of the decomposed kernel
    int local_top = conv_horizontal(img, channel_count, start, end, offset);

    // In order to normalize the batch, we need the values distribution from ALL the processes
    int global_top = local_top;

    for (int p = 0; p < n_processes; p++)
        if (p != rank)
            MPI_Send(&local_top, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

    for (int p = 0; p < n_processes; p++)
        if (p != rank)
        {
            MPI_Recv(&local_top, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_top = fmax(global_top, local_top);
        }

    // Normalizing the batch using the widest range
    normalize_batch(img, channel_count, global_top, start, end, offset);

    // Applying deptwise encoding to the image, incresing the number of channels by 'channel_multiplier'
    conv_depthwise_encode(img, channel_count * channel_multiplier, start, end, offset);
    // Compressing the array back into a 3-channel image
    conv_depthwise_decode(img, channel_count * channel_multiplier, start, end, offset);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    char *in_name = argv[1];
    char *out_name = argv[2];
    iterations = atoi(argv[3]);
    int channel_multiplier = atoi(argv[4]);

    if (rank == 0)
    {

        Channels **img = read_image_pnm(in_name, &width, &height);
        printf("%d %d\n", width, height);
        int start, end;

        for (int p = 1; p < n_processes; p++)
        {
            MPI_Send(&width, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(&height, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

            start = p * ceil((double)height / n_processes);
            end = fmin(height, (p + 1) * ceil((double)height / n_processes));

            // Send the padded parts of the image to each process
            for (int j = start - iterations; j < end + iterations; j++)
                if (j < height)
                {
                    MPI_Send(pack_channel(img[j], width + 2, 0), width + 2, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD);
                    MPI_Send(pack_channel(img[j], width + 2, 1), width + 2, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD);
                    MPI_Send(pack_channel(img[j], width + 2, 2), width + 2, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD);
                }
        }

        start = 0 * ceil((double)height / n_processes);
        end = fmin(height, (0 + 1) * ceil((double)height / n_processes) + iterations);
        int size = end - start;

        // Master will also process it's part of the image
        Channels **img0 = new_channel_array(size + 2 * iterations, width);

        for (int j = 0; j < size; j++)
            img0[j + iterations] = img[j];

        for (int i = 0; i < iterations; i++)
            conv_separable(img0, channel_multiplier, start, end, i);

        for (int j = 0; j < size; j++)
            img[j] = img0[j + iterations];

        // After the convolution is done, gather back the parts
        for (int i = 1; i < n_processes; i++)
        {
            start = i * ceil((double)height / n_processes);
            end = fmin(height, (i + 1) * ceil((double)height / n_processes));

            for (int j = start; j < end; j++)
            {
                unsigned char *red = malloc(width * sizeof(unsigned char));
                unsigned char *green = malloc(width * sizeof(unsigned char));
                unsigned char *blue = malloc(width * sizeof(unsigned char));

                MPI_Recv(red, width, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(green, width, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(blue, width, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                img[j] = unpack_rgb_channels(red, green, blue, width);
                free(red);
                free(green);
                free(blue);
            }
        }
        write_image_pnm(img, out_name, width, height);
    }
    else
    {
        MPI_Recv(&width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&height, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int start, end;
        start = rank * ceil((double)height / n_processes);
        end = fmin(height, (rank + 1) * ceil((double)height / n_processes));

        int size = end - start;

        // Each slave process gathers it's part of channels from master
        Channels **img = (Channels **)calloc(size + 2 * iterations, sizeof(Channels *));
        for (int j = 0; j < size + 2 * iterations; j++)
        {
            unsigned char *red = malloc(width * sizeof(unsigned char));
            unsigned char *green = malloc(width * sizeof(unsigned char));
            unsigned char *blue = malloc(width * sizeof(unsigned char));

            if (!(rank == n_processes - 1 && j >= size + iterations))
            {
                MPI_Recv(red, width + 2, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(green, width + 2, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(blue, width + 2, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            img[j] = unpack_rgb_channels(red, green, blue, width);
            free(red);
            free(green);
            free(blue);
        }

        /* There is no more communication at this point, each process can convolve it's padded 
            part of the image agnostic of the number of iterations.
        */
        for (int i = 0; i < iterations; i++)
            conv_separable(img, channel_multiplier, start, end, i);

        // Send back the processed part of the image back to master
        for (int j = iterations; j < size + iterations; j++)
        {
            MPI_Send(pack_channel(img[j], width, 0), width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
            MPI_Send(pack_channel(img[j], width, 1), width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
            MPI_Send(pack_channel(img[j], width, 2), width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}
