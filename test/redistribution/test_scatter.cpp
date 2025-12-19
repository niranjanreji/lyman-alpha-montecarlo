/*
TEST_SCATTER.CPP / NIRANJAN REJI
*/

#include <fstream>
#include <iostream>
#include <cassert>

#include "../../src/common.h"
#include "../../src/grid.cpp"
#include "../../src/physics.cpp"
using namespace std;

int main(int argc, char** argv) {
    // initialize test grid (1 voxel)
    Grid *g = new Grid();
    
    g->sqrt_temp.resize(1);
    g->vx.resize(1);
    g->vy.resize(1);
    g->vz.resize(1);

    // tunable scatter test parameters
    int num_samples = 3000000;        // no. of samples per input x value
    double T = 100.0;                    // temperature for tests
    double sqrt_T = sqrt(T);
    bool recoil = false;
    bool isotropic = true;

    g->sqrt_temp[0] = sqrt_T;
    g->vx[0] = 0; g->vy[0] = 0; g->vz[0] = 0;

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

            vector<double> mom_x, mom_y, mom_z;
            mom_x.resize(1); mom_y.resize(1); mom_z.resize(1);

            scatter(phot, *g, mom_x, mom_y, mom_z, rng, recoil, isotropic);

            fo << phot.x << "\n";
        }

        fo.close();
        cout << "Wrote " << fname << " (" << num_samples << " samples for xin = " << x_in << ")\n"; 
    }

    cout << "Done.\n";
    return 0;
}