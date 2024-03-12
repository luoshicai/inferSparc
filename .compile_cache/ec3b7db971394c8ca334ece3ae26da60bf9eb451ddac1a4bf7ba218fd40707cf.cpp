
        #include <iostream>
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <cmath>
        #include <functional>
        #include <tuple>
        #include <vector>
        #include <numeric>
        #include <chrono>

        using namespace std;

        int int_ceil(int x, int y){
            return (x - 1) / y + 1;
        }

        extern "C" void func(float* dense, float* sparse, int* masks, int *columns){
            int bm_m   = 768/128;
            int mcol_k = 768/4;
            int mcol_k_p = int_ceil(768,4);
            int m_fixed = 4;

            std::vector<float> v(m_fixed, 0);
            std::vector<int> vx(m_fixed, 0);
            float max=0, total=0;

            std::vector<size_t> indices(v.size());
            std::iota(indices.begin(), indices.end(), 0);

            for(int bm_i=0; bm_i<bm_m; bm_i++){
                int t_bm_i   = bm_i*128*768;
                for(int mcol_i=0; mcol_i<mcol_k; mcol_i++){
                    int t_mcol_i = mcol_i*4;
                    max = 0;

                    std::vector<int> cols_max;
                    cols_max.resize(m_fixed, 0);
                    std::vector<int> masks_max;
                    masks_max.resize(128*4, 0);

                    for(int col_i=0; col_i<4; col_i++){
                        vx[0]=col_i;
                        for(int col_j=col_i+1; col_j<4; col_j++){
                            vx[1]=col_j;
                            for(int col_k=col_j+1; col_k<4; col_k++){
                                vx[2]=col_k;
                                for(int col_w=col_k+1; col_w<4; col_w++){
                                    vx[3]=col_w;
                                    total=0;

                                    std::vector<int> mask(128*4, 0);

                                    for(int row_i=0; row_i<128; row_i++){
                                        int t_row_i  = row_i*768;
                                        v[0]=dense[t_bm_i + t_row_i + t_mcol_i + col_i];
                                        v[1]=dense[t_bm_i + t_row_i + t_mcol_i + col_j];
                                        v[2]=dense[t_bm_i + t_row_i + t_mcol_i + col_k];
                                        v[3]=dense[t_bm_i + t_row_i + t_mcol_i + col_w];

                                        std::partial_sort(indices.begin(), indices.begin() + 2, indices.end(), [&](size_t A, size_t B) {
                                                    return v[A] > v[B];});

                                        for(int id=0; id<2; id++){
                                            total += dense[t_bm_i + t_row_i + t_mcol_i + vx[indices[id]]];

                                            mask[row_i*4 + vx[indices[id]]] = 1;
                                        }
                                    }

                                    if(total>max){
                                        max = total;
                                        std::copy(mask.begin(), mask.end(), masks_max.begin());

                                        std::sort(vx.begin(), vx.begin() + m_fixed);
                                        std::copy(vx.begin(), vx.end(), cols_max.begin());
                                    }
                                }
                            }
                        }
                    }

                    for(int i=0; i<128; i++){
                        for(int j=0; j<4; j++){
                            int drop = masks_max[i*4 + j];
                            masks[t_bm_i  + i*768+ t_mcol_i + j]  = drop;
                            sparse[t_bm_i + i*768+ t_mcol_i + j] *= drop;
                        }
                    }
                    for(int i=0; i<m_fixed; i++){
                        columns[bm_i*768 + mcol_i*m_fixed + i] =
                        cols_max[i];
                    }
                }
            }

            int remainder = 768%4;

            if (remainder>0){
                int d_rows=128, d_cols=remainder;

                if(remainder<3){
                    for(int i=0; i<768; i++){
                        for(int j=768-remainder; j<768; j++){
                            masks[i*768+j] = 1;
                        }
                    }
                    for(int bm_i=0; bm_i<bm_m; bm_i++){
                        for(int j=0; j<m_fixed; j++){
                            columns[bm_i*768 + mcol_k*m_fixed + j] = j;
                        }
                    }
                } else if(remainder==3){
                    v[3] = -1;
                    for(int bm_i=0; bm_i<bm_m; bm_i++){
                        int t_bm_i   = bm_i*128*768;
                        for(int mcol_i=mcol_k; mcol_i<mcol_k_p; mcol_i++){
                            max = 0;
                            int t_mcol_i = mcol_i*4;

                            std::vector<int> cols_max(m_fixed, 0);
                            std::vector<int> masks_max(128*remainder, 0);

                            for(int col_i=0; col_i<remainder; col_i++){
                                vx[0]=col_i;
                                for(int col_j=col_i+1; col_j<remainder; col_j++){
                                    vx[1]=col_j;
                                    for(int col_k=col_j+1; col_k<remainder; col_k++){
                                        vx[2]=col_k;
                                        total=0;
                                        std::vector<int> mask(128*remainder, 0);

                                        for(int row_i=0; row_i<128; row_i++){
                                            int t_row_i  = row_i*768;
                                            v[0]=dense[t_bm_i + t_row_i + t_mcol_i + col_i];
                                            v[1]=dense[t_bm_i + t_row_i + t_mcol_i + col_j];
                                            v[2]=dense[t_bm_i + t_row_i + t_mcol_i + col_k];

                                            std::partial_sort(indices.begin(), indices.begin() + 2, indices.end(), [&](size_t A, size_t B) {
                                                        return v[A] > v[B]; });

                                            for(int id=0; id<2; id++){
                                                total += dense[t_bm_i + t_row_i + t_mcol_i + vx[indices[id]]];

                                                mask[row_i*remainder + vx[indices[id]]] = 1;
                                            }
                                        }

                                        if(total>max){
                                            max = total;
                                            std::copy(mask.begin(), mask.end(), masks_max.begin());

                                            std::sort(vx.begin(), vx.begin() + remainder);//m_fixed
                                            std::copy(vx.begin(), vx.end(), cols_max.begin());
                                        }
                                    }
                                }
                            }

                            for(int i=0; i<128; i++){
                                for(int j=0; j<remainder; j++){
                                    int drop = masks_max[i*remainder + j];

                                    masks[t_bm_i  + i*768+ t_mcol_i + j]  = drop;
                                    sparse[t_bm_i + i*768+ t_mcol_i + j] *= drop;
                                }
                            }
                            for(int i=0; i<remainder; i++){
                                columns[bm_i*768 + mcol_i*m_fixed + i] =
                                cols_max[i];
                            }
                        }
                    }
                } else {
                    for(int bm_i=0; bm_i<bm_m; bm_i++){
                        int t_bm_i   = bm_i*128*768;
                        for(int mcol_i=mcol_k; mcol_i<mcol_k_p; mcol_i++){
                            max = 0;
                            int t_mcol_i = mcol_i*4;

                            std::vector<int> cols_max(m_fixed, 0);
                            std::vector<int> masks_max(128*remainder, 0);

                            for(int col_i=0; col_i<remainder; col_i++){
                                vx[0]=col_i;
                                for(int col_j=col_i+1; col_j<remainder; col_j++){
                                    vx[1]=col_j;
                                    for(int col_k=col_j+1; col_k<remainder; col_k++){
                                        vx[2]=col_k;
                                        for(int col_w=col_k+1; col_w<remainder; col_w++){
                                            vx[3]=col_w;
                                            total=0;
                                            std::vector<int> mask(128*remainder, 0);

                                            for(int row_i=0; row_i<128; row_i++){
                                                int t_row_i  = row_i*768;
                                                v[0]=dense[t_bm_i + t_row_i + t_mcol_i + col_i];
                                                v[1]=dense[t_bm_i + t_row_i + t_mcol_i + col_j];
                                                v[2]=dense[t_bm_i + t_row_i + t_mcol_i + col_k];
                                                v[3]=dense[t_bm_i + t_row_i + t_mcol_i + col_w];

                                                std::partial_sort(indices.begin(), indices.begin() + 2, indices.end(), [&](size_t A, size_t B) {
                                                            return v[A] > v[B]; });

                                                for(int id=0; id<2; id++){
                                                    total += dense[t_bm_i + t_row_i + t_mcol_i + vx[indices[id]]];

                                                    mask[row_i*remainder + vx[indices[id]]] = 1;
                                                }
                                            }

                                            if(total>max){
                                                max = total;
                                                std::copy(mask.begin(), mask.end(), masks_max.begin());

                                                std::sort(vx.begin(), vx.begin() + m_fixed);
                                                std::copy(vx.begin(), vx.end(), cols_max.begin());
                                            }
                                        }
                                    }
                                }
                            }

                            for(int i=0; i<128; i++){
                                for(int j=0; j<remainder; j++){
                                    int drop = masks_max[i*remainder + j];

                                    masks[t_bm_i  + i*768+ t_mcol_i + j]  = drop;
                                    sparse[t_bm_i + i*768+ t_mcol_i + j] *= drop;
                                }
                            }
                            for(int i=0; i<m_fixed; i++){
                                columns[bm_i*768 + mcol_i*m_fixed + i] =
                                cols_max[i];
                            }
                        }
                    }
                }
            }
        }
        