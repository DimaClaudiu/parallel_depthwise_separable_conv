#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../Utils/utils.h"

int n_threads = 4;
int width;
int height;
int global_top;
int channel_multiplier;
Channels **img;
pthread_mutex_t mutex_top;

// Standardizes a batch 1 image into the range 0-255
void *normalize_batch(void *var)
{
    float upscale_factor = 255.f / global_top;
    int i, j, c;

    int thread_id = *(int *)var;
    unsigned long start, end;
    start = thread_id * ceil((double)height / n_threads);
    end = fmin(height, (thread_id + 1) * ceil((double)height / n_threads));

    for (i = start; i < end; i++)
        for (j = 1; j < width + 1; j++)
            for (c = 0; c < channel_count; c++)
                img[i][j].channel[c] = clamp_to_byte(upscale_factor * img[i][j].channel[c]);
}

// Applies the vertical part of the spatial sepratable convolution
void *conv_vertical(void *var)
{
    int thread_id = *(int *)var;
    unsigned long start, end;
    start = thread_id * ceil((double)height / n_threads);
    end = fmin(height, (thread_id + 1) * ceil((double)height / n_threads));

    int i, j, m, c, k;
    float K[channel_count] = {1.f / channel_count, 2.f / channel_count, 1.f / channel_count};

    // The array is bordered with 0
    Channels **bordered_img = new_channel_array(height + 2, width + 2);

    for (i = fmax(0, (int)start - 2); i < fmin(height, (int)end + 2); i++)
        for (j = 0; j < width; j++)
            bordered_img[i][j + 1] = img[i][j];

    for (i = start; i <= end; i++)
        for (j = 1; j < width + 1; j++)
        {
            if (i == height)
                continue;
            float *final_pixel = calloc(channel_count, sizeof(float));
            for (m = -1, k = 0; m <= 1; m++, k++)
                for (c = 0; c < channel_count; c++)
                    final_pixel[c] += bordered_img[i + 1 + m][j].channel[c] * K[k];

            for (c = 0; c < channel_count; c++)
                img[i][j - 1].channel[c] = clamp_to_byte(final_pixel[c]);

            free(final_pixel);
        }
}

// Applies the horizonal part of the spatial sepratable convolution
void *conv_horizontal(void *var)
{
    int thread_id = *(int *)var;
    unsigned long start, end;
    start = thread_id * ceil((double)height / n_threads);
    end = fmin(height, (thread_id + 1) * ceil((double)height / n_threads));

    int i, j, n, c, k;
    int top = 0;
    float K[channel_count] = {-1.f / channel_count, 0 / channel_count, 1.f / channel_count};

    Channels **bordered_img = new_channel_array(height + 2, width + 2);

    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
            bordered_img[i][j + 1] = img[i][j];

    for (i = start; i <= end; i++)
        for (j = 1; j < width + 1; j++)
        {
            if (i == height)
                continue;
            float *final_pixel = calloc(channel_count, sizeof(float));
            for (n = -1, k = 0; n <= 1; n++, k++)
                for (c = 0; c < channel_count; c++)
                    final_pixel[c] += bordered_img[i + 1][j + n].channel[c] * K[k];

            for (c = 0; c < channel_count; c++)
            {
                img[i][j - 1].channel[c] = clamp_to_byte(final_pixel[c]);
                // Also keep in mind the top range of the distribution here to avoid another traversal
                top = fmax(top, img[i][j - 1].channel[c]);
            }

            free(final_pixel);
        }

    pthread_mutex_lock(&mutex_top);
    if (top > global_top)
        global_top = top;
    pthread_mutex_unlock(&mutex_top);
}

/* Applies a depthwise convolution with a random kernel, increasing the number of channels
to the specified amount. Must be a multiple of the original arrays number of channels! */
void *conv_depthwise_encode(void *var)
{
    // This kernel should be learned by our network, but here we will randomly generate it
    float *K = get_kernel(42);
    int i, j, final_pixel, c;

    int thread_id = *(int *)var;
    unsigned long start, end;
    start = thread_id * ceil((double)height / n_threads);
    end = fmin(height, (thread_id + 1) * ceil((double)height / n_threads));

    // Pools the channels with a stride of 'channel_count'
    for (int channel_id = channel_count; channel_id < channel_count * channel_multiplier; channel_id++)
    {
        for (i = start; i < end; i++)
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
void *conv_depthwise_decode(void *var)
{
    int i, j, red, green, blue, c;

    int thread_id = *(int *)var;
    unsigned long start, end;
    start = thread_id * ceil((double)height / n_threads);
    end = fmin(height, (thread_id + 1) * ceil((double)height / n_threads));

    for (i = 1; i < height; i++)
        for (j = 1; j < width + 1; j++)
        {
            red = 0;
            green = 0;
            blue = 0;

            // This has the effect of averaging out all the dimensions into 3 colors
            // An additional kernel could be used for more complex outputs
            for (c = 0; c < channel_count * channel_multiplier; c += channel_count)
            {
                red += img[i][j].channel[c + 0];
                green += img[i][j].channel[c + 1];
                blue += img[i][j].channel[c + 2];
            }
            img[i][j].channel[0] = clamp_to_byte(red / channel_multiplier);
            img[i][j].channel[1] = clamp_to_byte(green / channel_multiplier);
            img[i][j].channel[2] = clamp_to_byte(blue / channel_multiplier);
        }
}

/* Applies the depthwise separable convolution to the given image.
    - channel_multiplier: Applies a polling step the the array, extending the number of channels
by the given amount.
*/
void conv_separable(int iterations)
{
    int i;
    pthread_mutex_init(&mutex_top, NULL);
    pthread_t tid[n_threads];
    int thread_id[n_threads];
    for (i = 0; i < n_threads; i++)
        thread_id[i] = i;

    for (int j = 0; j < iterations; j++)
    {
        // First we apply the vertical kernel
        for (i = 0; i < n_threads; i++)
            pthread_create(&(tid[i]), NULL, conv_vertical, &(thread_id[i]));

        for (i = 0; i < n_threads; i++)
            pthread_join(tid[i], NULL);

        // The applying the horizonal part of the decomposed kernel
        for (i = 0; i < n_threads; i++)
            pthread_create(&(tid[i]), NULL, conv_horizontal, &(thread_id[i]));

        for (i = 0; i < n_threads; i++)
            pthread_join(tid[i], NULL);

        // Normalizing the batch using the widest range
        for (i = 0; i < n_threads; i++)
            pthread_create(&(tid[i]), NULL, normalize_batch, &(thread_id[i]));

        for (i = 0; i < n_threads; i++)
            pthread_join(tid[i], NULL);

        // Applying deptwise encoding to the image, incresing the number of channels by 'channel_multiplier'
        for (i = 0; i < n_threads; i++)
            pthread_create(&(tid[i]), NULL, conv_depthwise_encode, &(thread_id[i]));

        for (i = 0; i < n_threads; i++)
            pthread_join(tid[i], NULL);

        // Compressing the array back into a 3-channel image
        for (i = 0; i < n_threads; i++)
            pthread_create(&(tid[i]), NULL, conv_depthwise_decode, &(thread_id[i]));

        for (i = 0; i < n_threads; i++)
            pthread_join(tid[i], NULL);
    }
}

int main(int argc, char *argv[])
{
    n_threads = atoi(argv[1]);
    char *in_name = argv[2];
    char *out_name = argv[3];
    int iterations = atoi(argv[4]);
    channel_multiplier = atoi(argv[5]);

    img = read_image_pnm(in_name, &width, &height);

    conv_separable(iterations);

    write_image_pnm(img, out_name, width, height);

    return 0;
}
