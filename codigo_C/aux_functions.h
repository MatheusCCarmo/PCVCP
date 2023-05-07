
#ifndef AUX_FUNCTIONS_H
#define AUX_FUNCTIONS_H

int contains(int *arr, int arr_len, int elem)
{
    for (int i = 0; i < arr_len; i++)
    {
        if (arr[i] == elem)
        {
            return 1;
        }
    }
    return 0;
}

int get_route_length(int *route)
{
    int length = 0;
    while (route[length] != -1)
    {
        length++;
    }
    return length;
}

#endif
