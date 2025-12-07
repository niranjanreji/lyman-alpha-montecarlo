/*
TEST_SCATTER.CPP / NIRANJAN REJI
*/

#include <fstream>
#include <iostream>
#include <cassert>

#include "../../routines/common.h"
#include "../../routines/model.cpp"
#include "../../routines/physics.cpp"
using namespace std;

int main(int argc, char** argv) {
    // initialize test grid (1 voxel)
    g_grid.sqrt_T.resize(1);
    g_grid.vx.resize(1);
    g_grid.vy.resize(1);
    g_grid.vz.resize(1);

    // tunable scatter test parameters
    int num_samples = 3000000;        // no. of samples per input x value
    double T = 100.0;                    // temperature for tests
    double sqrt_T = sqrt(T);
    bool recoil = false;
    bool isotropic = true;

    g_grid.sqrt_T[0] = sqrt_T;
    g_grid.vx[0] = 0; g_grid.vy[0] = 0; g_grid.vz[0] = 0;

    vector<double> xins = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0};

    xso::rng rng(0);

    for (double x_in : xins) {
        string fname = "out_xin_" + to_string(int(x_in)) + ".dat";
        ofstream fo(fname);

        for (int i = 0; i < num_samples; ++i) {
            Photon phot;
            phot.x = x_in;
            phot.dir_x = 1;
            phot.dir_y = 0; phot.dir_z = 0;
            phot.pos_x = 0; phot.pos_y = 0; phot.pos_z = 0;

            scatter(phot, 0, 0, 0, rng, recoil, isotropic);

            fo << phot.x << "\n";
        }

        fo.close();
        cout << "Wrote " << fname << " (" << num_samples << " samples for xin = " << x_in << ")\n"; 
    }

    cout << "Done.\n";
    return 0;
}