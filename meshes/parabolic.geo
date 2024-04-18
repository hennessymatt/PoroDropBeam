// Gmsh project created on Thu Feb 24 23:03:27 2022

// Domain characteristics
L = 1.0;

// Beam height
// H = 75e-6 / 17e-3;


// H = 0.1;
// h = H;
// eb = H / 20;
// ed = h / 20;
// ec = ed / 20;

// H = 0.05;
// h = H;
// eb = 2.5e-3;
// ed = 5e-3;
// ec = 5e-4;

// H = 0.01;
// h = H;
// eb = H / 10;
// ed = h / 10;
// ec = ed / 10;

h = 0.01;
H = 0.1;

eb = H / 5;
ed = h / 20;
ec = ed / 2;

N = 100;
//h = 1.0 / N;

// Define mesh for the drop

For i In {0:N}
x = i / N;
h_f = 4 * h * x * (1 - x);
e = (ed - ec) * h_f / h + ec;
Point(i) = {x, h_f, 0, e};
EndFor
Point(N+1) = {L/2, 0, 0, ed};

For i In {1:N}
Line(i) = {i, i-1};
EndFor
Line(N+1) = {0, N+1};
Line(N+2) = {N+1, N};

Curve Loop(1) = {1:N+2};

Plane Surface(1) = {1};

// Define mesh for the beam
Point(N+2) = {0, -H, 0, eb};
Point(N+3) = {L, -H, 0, eb};

Line(N+3) = {N+2, 0};
Line(N+4) = {N, N+3};
Line(N+5) = {N+3, N+2};


Curve Loop(2) = {N+3, N+1, N+2, N+4, N+5};

Plane Surface(2) = {2};

// Create physical surfaces

// free surface
Physical Curve(1) = {1:N};
// substrate
Physical Curve(2) = {N+1, N+2};
// wall
Physical Curve(3) = {N+3};
// bulk (drop)
Physical Surface(4) = {1};
// bulk (beam)
Physical Surface(5) = {2};

// Save mesh

// Mesh 2;
// Save "test2.msh";
