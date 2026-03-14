#include <fstream>
#include <iostream>
#include "common.h"
#include "rt_definitions.h"



using namespace std;

int main(int argc, char* argv[]) {

    /* number of photons to test for each input frequency */
    int num_samples = 3000000;
    
    Grid* grid = init_grid(user_sources);
    Photons* p = new Photons();

    build_fields(grid, user_density, user_temperature, user_velocity);

    /* input frequencies for redistribution test */
    vector<double> xins = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0};

    xso::rng rng(0);

    for (double x_in : xins) {
        string fname = "out_xin_" + to_string(int(x_in)) + ".dat";
        string fpath = "output/" + fname;
        ofstream fo(fpath);

        for (int i = 0; i < num_samples; ++i) {
            Photon phot;
            phot.x = x_in;
            phot.dir_x = 1;
            phot.dir_y = 0; phot.dir_z = 0;
            phot.pos_x = 0; phot.pos_y = 0; phot.pos_z = 0;

            scatter(phot, *grid, rng);
            fo << phot.x << "\n";
        }

        fo.close();
        cout << "Wrote " << fname << " (" << num_samples << " samples for xin = " << x_in << ")\n"; 
    }

    cout << "Done.\n";

    return 0;
}