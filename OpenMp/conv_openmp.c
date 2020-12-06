#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../Utils/utils.h"

int n_threads = 4;
int width;
int height;
int iterations;

// Standardizes a batch 1 image into the range 0-255
Channels **normalize_batch(Channels **img, int num_channels, int top)
{
    float upscale_factor = 255.f / top;
    int i, j, c;

#pragma omp parallel for private(i, j, c) collapse(3) shared(img)
    for (i = 1; i < height; i++)
        for (j = 1; j < width + 1; j++)
            for (c = 0; c < num_channels; c++)
                img[i][j].channel[c] = clamp_to_byte(upscale_factor * img[i][j].channel[c]);

    return img;
}

// Applies the vertical part of the spatial sepratable convolution
void conv_vertical(Channels **img, int num_channels)
{
    int i, j, m, c, k;
    float K[channel_count] = {1.f / channel_count, 2.f / channel_count, 1.f / channel_count};

    // The array is bordered with 0
    Channels **bordered_img = new_channel_array(height + 1, width + 1);

    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
            bordered_img[i][j + 1] = img[i][j];

#pragma omp parallel for private(i, j, m, c, k) collapse(2) shared(img, bordered_img)
    for (i = 1; i < height; i++)
        for (j = 1; j < width + 1; j++)
        {
            float *final_pixel = calloc(num_channels, sizeof(float));
            for (m = -1, k = 0; m <= 1; m++, k++)
                for (c = 0; c < num_channels; c++)
                    final_pixel[c] += bordered_img[i + m][j].channel[c] * K[k];

            for (c = 0; c < num_channels; c++)
                img[i][j - 1].channel[c] = clamp_to_byte(final_pixel[c]);

            free(final_pixel);
        }
}

// Applies the horizonal part of the spatial sepratable convolution
int conv_horizontal(Channels **img, int num_channels)
{
    int i, j, n, c, k;
    int top = 0;
    float K[channel_count] = {-1.f / channel_count, 0 / channel_count, 1.f / channel_count};

    Channels **bordered_img = new_channel_array(height + 1, width + 1);

    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
            bordered_img[i][j + 1] = img[i][j];

#pragma omp parallel for private(i, j, n, c, k) collapse(2) shared(img, bordered_img)
    for (i = 1; i < height; i++)
        for (j = 1; j < width + 1; j++)
        {
            float *final_pixel = calloc(num_channels, sizeof(float));
            for (n = -1, k = 0; n <= 1; n++, k++)
                for (c = 0; c < num_channels; c++)
                    final_pixel[c] += bordered_img[i][j + n].channel[c] * K[k];

            for (c = 0; c < num_channels; c++)
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
void conv_depthwise_encode(Channels **img, int num_channels)
{
    // This kernel should be learned by our network, but here we will randomly generate it
    float *K = get_kernel(42);
    int i, j, final_pixel, c;

    // Pools the channels with a stride of 'channel_count'
#pragma omp parallel for private(i, j, final_pixel, c) collapse(3) shared(img)
    for (int channel_id = channel_count; channel_id < num_channels; channel_id++)
    {
        for (i = 1; i < height; i++)
            for (j = 1; j < width + 1; j++)
            {
                final_pixel = 0;
                for (c = 0; c < channel_count; c++)
                    final_pixel += img[i][j].channel[channel_id % channel_count] * K[c];

                img[i][j].channel[channel_id] = clamp_to_byte(final_pixel);
            }
    }
    free(K);
}

/* Applies the depthwise convolution with a static kernel, transforming the array back
into a 3-channel image. */
void conv_depthwise_decode(Channels **img, int num_channels)
{
    int multiplier = num_channels / channel_count;
    int i, j, red, green, blue, c;

#pragma omp parallel for private(i, j, c) collapse(2) shared(img)
    for (i = 1; i < height; i++)
        for (j = 1; j < width + 1; j++)
        {
            red = 0;
            green = 0;
            blue = 0;

            // This has the effect of averaging out all the dimensions into 3 colors
            // An additional kernel could be used for more complex outputs
            for (c = 0; c < num_channels; c += channel_count)
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
    - channel_multiplier: Applies a polling step the the array, extending the number of channels
by the given amount.
*/
void conv_separable(Channels **img, int channel_multiplier)
{
    omp_set_num_threads(n_threads);

    // First we apply the vertical kernel
    conv_vertical(img, channel_count);

    // The applying the horizonal part of the decomposed kernel
    int top = conv_horizontal(img, channel_count);

    // Normalizing the batch using the widest range
    normalize_batch(img, channel_count, top);

    // Applying deptwise encoding to the image, incresing the number of channels by 'channel_multiplier'
    conv_depthwise_encode(img, channel_count * channel_multiplier);
    // Compressing the array back into a 3-channel image
    conv_depthwise_decode(img, channel_count * channel_multiplier);
}

int main(int argc, char *argv[])
{
    n_threads = atoi(argv[1]);
    char *in_name = argv[2];
    char *out_name = argv[3];
    iterations = atoi(argv[4]);
    int channel_multiplier = atoi(argv[5]);

    Channels **img = read_image_pnm(in_name, &width, &height);

    for (int i = 0; i < iterations; i++)
        conv_separable(img, channel_multiplier);

    write_image_pnm(img, out_name, width, height);

    return 0;
}
