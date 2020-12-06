#ifndef UTILS_H_
#define UTILS_H_

#define LAYER_HEIGHT 32
#define channel_count 3

typedef struct
{
    unsigned char channel[LAYER_HEIGHT];
} Channels;

unsigned char clamp_to_byte(float byte);
Channels **read_image_pnm(char *filename, int *width, int *height);
void write_image_pnm(Channels **img, char *filename, int width, int height);
int get_range(Channels **img, int width, int height);
float *get_kernel(int kernel_id);
Channels **new_channel_array(int height, int width);

#endif // UTILS_H_