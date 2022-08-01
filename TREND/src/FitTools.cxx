#include "FitTools.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iomanip>
#include <time.h>

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FitTools::FitTools():
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Class constructor. Main task is tuning the PORT_i interfaces handling
//  the minimisation.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
_xPORT( MAX_PARAMETERS ) // There are up to MAX_PARAMETERS parameters to minimise.
{


  // Initialisation of antennas times and positions
  //===
  memset( _Xa, 0x0, (N_ANTENNA_DATA*MAX_ANTENNA+ADD_USER_SPACE)
    *sizeof(double) );
  _Na = 0;

  // Initialisation of error matrix.
  //===
  memset( _errorM,  0x0, MAX_PARAMETERS*MAX_ANTENNA*sizeof( double ) );
  memset( _errorII, 0x0, MAX_PARAMETERS*sizeof( double ) );
  _sigma_t = 0.0;

  // Initialisation of gchi2 vars
  //===
  _gchi2_algorithm = 204;
  _lambda_153[ 0 ] = 0.0;
  _p_lambda_153 = _lambda_153+1;

  // Minimiser tuning
  //===
  _xPORT.max_func_evals() = 10000;
  _xPORT.max_iterations() = 10000;
  _xPORT.printing_off(); // Mute the minimiser, comment this for debug

  // Default fit settings
  //===
  _fitModel      = PLAN_WAVE;
  _fixedSource   = false;
  _fixedSpeed    = false;
  _fixedZs       = false;
  _cr            = 1.0;
  _computeErrors = true;

  // Default parameter initialisation
  //===
  _theta = _phi = 0.0;
  _x_xmax = _y_xmax = _z_xmax = 0.0;
  _cerenkov_width = _amp_corr = 0.0;

  // Initialise random engine
  //===
  srand48( time( NULL ) );
  _randn_parity = false;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FitTools::~FitTools()
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::_initFit()
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define MAX_R 1e10
{
  // Reset constrains
  //===
  for ( int i = 0; i < _xPORT.N(); i++ )
  {
    _xPORT.B( 1, i+1 ) = -MAX_R;
    _xPORT.B( 2, i+1 ) =  MAX_R;
  }

  if ( _fitModel == PLAN_WAVE )
  {
    _nParameters = 2;

    // Connect the objective function to minimise and its gradient
    //===
    _xPORT.f = &PWF_function_F;
    _xPORT.g = &PWF_function_G;

    // Set bounding: 0 <= theta <= pi
    //===
    _xPORT.B( 1, 1 ) = 90.0*M_PI/180;    // theta parameter (1st) min value (1) is 0.0
    _xPORT.B( 2, 1 ) = 180.0*M_PI/180;   // theta parameter (1st) max value (2) is pi
    _xPORT.B( 1, 2 ) = 0.0;    // phi parameter (2nd) min value (1) is 0.0
    _xPORT.B( 2, 2 ) = 2*M_PI; // phi parameter (2nd) max value (2) is 2*pi

    //Debug
    // _xPORT.B( 1, 1 ) = 92.92*M_PI/180;  // theta parameter (1st) min value (1)
    // _xPORT.B( 2, 1 ) = 92.92*M_PI/180;  // theta parameter (1st) max value (2)
    // _xPORT.B( 1, 2 ) = 180.*M_PI/180;    // phi parameter (2nd) min value (1)
    // _xPORT.B( 2, 2 ) = 180.*M_PI/180;    // phi parameter (2nd) max value (2)

    // Initialise fit parameters
    //===
    _xPORT.X( 1 ) = _theta;
    _xPORT.X( 2 ) = _phi;
    _xPORT.X( 3 ) = _cr;
  }
  else if (_fitModel == SPHERE_WAVE)
  {
    _nParameters = 4;

    // Connect the objective function to minimise and its gradient
    //===
    _xPORT.f = &SWF_function_F;
    _xPORT.g = &SWF_function_G;

    // Set bounding: 0 <= theta <= pi
    //===

    _xPORT.B( 1, 1 ) = (_theta_input-1.)*M_PI/180;  // theta parameter (1st) min value (1)
    _xPORT.B( 2, 1 ) = (_theta_input+1.)*M_PI/180;  // theta parameter (1st) max value (2)
    _xPORT.B( 1, 2 ) = (_phi_input-.1)*M_PI/180;    // phi parameter (2nd) min value (1)
    _xPORT.B( 2, 2 ) = (_phi_input+.1)*M_PI/180;    // phi parameter (2nd) max value (2)
    _xPORT.B( 1, 3) = -15.6e3 - 12.3e3/cos(_theta_input*M_PI/180);    // r parameter (3th) min value (1)
    _xPORT.B( 2, 3) = -6.1e3 - 15.4e3/cos(_theta_input*M_PI/180);     // r parameter (3th) max value (2)
    _xPORT.B( 1, 4) = -(-6.1e3 - 15.4e3/cos(_theta_input*M_PI/180));  // t parameter (3th) max value (2) similar to r !
    _xPORT.B( 2, 4) = 0;                                              // t parameter (3th) min value (1) similar to r !
    //std::cout<<"_xPORT.B( 1, 3) = "<<_xPORT.B( 1, 3)<<" _xPORT.B( 2, 3) = "<<_xPORT.B( 2, 3)<<std::endl;


    // //Debug
    // _xPORT.B( 1, 1 ) = 92.92*M_PI/180.;  // theta parameter (1st) min value (1)
    // _xPORT.B( 2, 1 ) = 92.92*M_PI/180.;  // theta parameter (1st) max value (2)
    // _xPORT.B( 1, 2 ) = M_PI;    // phi parameter (2nd) min value (1)
    // _xPORT.B( 2, 2 ) = M_PI;    // phi parameter (2nd) max value (2)
    // _xPORT.B( 1, 3) = 249870.0;                       // r parameter (3th) min value (1)
    // _xPORT.B( 2, 3) = 249870.0;                       // r parameter (3th) max value (2)
    // _xPORT.B( 1, 4) = -1.e6;                       // t xmax parameter (3th) min value (1)
    // _xPORT.B( 2, 4) = 1.e6;                        // t xmax parameter (3th) max value (2)

    // Initialise fit parameters
    //===
    _xPORT.X( 1 ) = _theta;
    _xPORT.X( 2 ) = _phi;
    _xPORT.X( 3 ) = _r_xmax;
    _xPORT.X( 4 ) = _ts;
    _xPORT.X( 5 ) = _cr;

  }
  else if (_fitModel == ADF)
  {
    _nParameters = 4;

    // Connect the objective function to minimise and its gradient
    //===
    _xPORT.f = &ADF_function_F;
    _xPORT.g = &ADF_function_G;

    // Set bounding: 0 <= theta <= pi
    //===
    _xPORT.B( 1, 1 ) = (_theta_input-2.)*M_PI/180;  // theta parameter (1st) min value (1)
    _xPORT.B( 2, 1 ) = (_theta_input+2.)*M_PI/180;  // theta parameter (1st) max value (2)
    _xPORT.B( 1, 2 ) = (_phi_input-1.)*M_PI/180;    // phi parameter (2nd) min value (1)
    _xPORT.B( 2, 2 ) = (_phi_input+1.)*M_PI/180;    // phi parameter (2nd) max value (2)
    _xPORT.B( 1, 3) = 0.1;                          // cerenkov width parameter (4th) min value (1)
    _xPORT.B( 2, 3) = 3.0;                          // cerenkov width (4th) max value (2)
    _xPORT.B( 1, 4) = 1.e6;                         // amplitude correction parameter (5th) min value (1)
    _xPORT.B( 2, 4) = 1.e10;                        // amplitude correction (5th) max value (2)

    //Debug
    // _xPORT.B( 1, 1 ) = 93.5*M_PI/180;  // theta parameter (1st) min value (1)
    // _xPORT.B( 2, 1 ) = 93.5*M_PI/180;  // theta parameter (1st) max value (2)
    // _xPORT.B( 1, 2 ) = 180.*M_PI/180;    // phi parameter (2nd) min value (1)
    // _xPORT.B( 2, 2 ) = 180.*M_PI/180;    // phi parameter (2nd) max value (2)
    // _xPORT.B( 1, 3) = 1.37;                          // cerenkov width parameter (4th) min value (1)
    // _xPORT.B( 2, 3) = 1.37;                          // cerenkov width (4th) max value (2)
    // _xPORT.B( 1, 4) = 9.0e7;                         // amplitude correction parameter (5th) min value (1)
    // _xPORT.B( 2, 4) = 9.0e7;                        // amplitude correction (5th) max value (2)


    // Initialise fit parameters
    //===
    _xPORT.X( 1 ) = _theta;
    _xPORT.X( 2 ) = _phi;
    _xPORT.X( 3 ) = _cerenkov_width;
    _xPORT.X( 4 ) = _amp_corr;
    _xPORT.X( 5 ) = _cr;

  }
  if ( !_fixedSpeed && (_fitModel == PLAN_WAVE))
  {
    //  Add wave speed as last parameter
    //===
    _nParameters++;

    // Set bounding: cr >= 0
    //===
    _xPORT.B( 1, _nParameters ) = 0.0; // Speed parameter min value is 0.0
  }
}
#undef MAX_R


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::_copyFitParameters()
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( _fitModel == PLAN_WAVE )
  {
    // Get fit parameters
    //===
    _theta = _xPORT.X( 1 );
    _phi   = _xPORT.X( 2 );
    _cr    = _xPORT.X( 3 );
  }
  else if ( _fitModel == SPHERE_WAVE )
  {
    // Get fit parameters
    //===
    _theta= _xPORT.X( 1 );
    _phi= _xPORT.X( 2 );
    _r_xmax= _xPORT.X( 3 );
    _ts    = _xPORT.X( 4 );
    _cr    = _xPORT.X( 5 );
  }
  else if ( _fitModel == ADF )
  {
    // Get fit parameters
    //===
    _theta = _xPORT.X( 1 );
    _phi   = _xPORT.X( 2 );
    _cerenkov_width= _xPORT.X( 3 );
    _amp_corr= _xPORT.X( 4 );
    _cr    = _xPORT.X( 5 );
  }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double& FitTools::xa( const int& n )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the nth antenna x position.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( n <= 0 ) || ( n > MAX_ANTENNA ) )
  {
    _dummyReal = 0.0;
    return _dummyReal;
  }
  else
    return _Xa[ N_ANTENNA_DATA*( n - 1 ) ];
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double& FitTools::ya( const int& n )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the nth antenna y position.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( n <= 0 ) || ( n > MAX_ANTENNA ) )
  {
    _dummyReal = 0.0;
    return _dummyReal;
  }
  else
    return _Xa[ N_ANTENNA_DATA*( n - 1 ) + 1 ];
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double& FitTools::za( const int& n )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the nth antenna z position.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( n <= 0 ) || ( n > MAX_ANTENNA ) )
  {
    _dummyReal = 0.0;
    return _dummyReal;
  }
  else
    return _Xa[ N_ANTENNA_DATA*( n - 1 ) + 2 ];
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double& FitTools::ta( const int& n )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the nth antenna signal arrival time.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( n <= 0 ) || ( n > MAX_ANTENNA ) )
  {
    _dummyReal = 0.0;
    return _dummyReal;
  }
  else
    return _Xa[ N_ANTENNA_DATA*( n - 1 ) + 3 ];
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double& FitTools::amplitude( const int& n )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the nth antenna amplitude.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( n <= 0 ) || ( n > MAX_ANTENNA ) )
  {
    _dummyReal = 0.0;
    return _dummyReal;
  }
  else
    return _Xa[ N_ANTENNA_DATA*( n - 1 ) + 4 ];
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double& FitTools::x_Xmax( const int& n )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the x xmax position.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( n <= 0 ) || ( n > MAX_ANTENNA ) )
  {
    _dummyReal = 0.0;
    return _dummyReal;
  }
  else
    return _Xa[ N_ANTENNA_DATA*( n - 1 ) + 5 ];
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double& FitTools::y_Xmax( const int& n )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the y xmax position.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( n <= 0 ) || ( n > MAX_ANTENNA ) )
  {
    _dummyReal = 0.0;
    return _dummyReal;
  }
  else
    return _Xa[ N_ANTENNA_DATA*( n - 1 ) + 6 ];
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double& FitTools::z_Xmax( const int& n )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the z xmax position.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( n <= 0 ) || ( n > MAX_ANTENNA ) )
  {
    _dummyReal = 0.0;
    return _dummyReal;
  }
  else
    return _Xa[ N_ANTENNA_DATA*( n - 1 ) + 7 ];
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::error( const int& i, const int& j )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the error propagation coefficient on parameter i resulting
//  from measurement j.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( j <= 0 ) || ( j > MAX_ANTENNA ) ||
       ( i <= 0 ) || ( i > MAX_PARAMETERS ) )
    return ( 0.0 );
  else
    return _errorM[ i-1 ][ j-1 ];
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::error( const int& i )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Reference to the error coefficient on parameter i for identical and
//  independently distributed errors on inputs with unit sigma. Multiply
//  by the common sigma value of input measurements to get the true
//  parameters error.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( ( i <= 0 ) || ( i > MAX_PARAMETERS ) )
    return( 0.0 );
  else
    return _errorII[ i-1 ];
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::fit( int na )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Perform a single fit with user suplied initial conditions. Returns the
//  fit chi2. In case of faillure -1 is returned. The fit best guess is
//  stored in fit parameters vector.
//
//  The number na of antenna can be provided as input argument. The current
//  value of _Na is assumed otherwise.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  // Update number of antennas if required
  //===
  if ( na > 0 )
    _Na = na;

  double t0, z0, amplitude0;
  if ( _fitModel == PLAN_WAVE || _fitModel==SPHERE_WAVE)
  {
    // Regularise the time origin
    //===
    t0 = this->ta( 1 );
    for ( int i = 2; i <= _Na; i++ )
      if ( this->ta( i ) < t0 )
        t0 = this->ta( i );
    for ( int i = 1; i <= _Na; i++ )
      this->ta( i )-= t0;
  }

  // Initialise according to fit model
  //===
  _initFit();

  // Do the fit
  //===
  double chi2 = -1;
  //if ( _xPORT.MNGB( &_Na, _Xa, NULL, _nParameters ) )
  if ( _xPORT.MNFB( &_Na, _Xa, NULL,  _nParameters ) )
  {
    chi2 = _xPORT.value_f();
    this->_copyFitParameters();
    //std::cout<<"New chi2 = "<<chi2<<" Theta = "<<_xPORT.X( 1 )*180./M_PI<<" Phi = "<<_xPORT.X( 2 )*180./M_PI<<std::endl;
  }

  if (_fitModel == PLAN_WAVE || _fitModel==SPHERE_WAVE)
  {
    // Restore the time origin
    //===
    for ( int i = 1; i <= _Na; i++ )
      this->ta( i )+= t0;
      if ( _fitModel == SPHERE_WAVE )
      {
        this->ts()+= t0;
        _xPORT.X( 4 )+= t0;
      }
  }

  // Normalise chi2
  //===
  if ( (_fitModel == PLAN_WAVE) && ( _sigma_t > 0.0 ) )
  {
    chi2 /= _sigma_t*_sigma_t;
  }
  return chi2;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::scan( int na )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Generic routine for scan like algorithms. The scan tries several fit with
//  various initial conditions. The minimum chi2 over all fits is returned.
//  In case of faillure -1 is returned. The fit best guess is stored as
//  parameter vector.
//
//  The number na of antenna can be provided as input argument. The current
//  value of _Na is assumed otherwise.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  // Update number of antennas if required
  //===
  if ( na > 0 )
    _Na = na;

  // Initialise according to fit model
  //===
  _initFit();

  // Route on specific scan algorithm
  //===
  double chi2;
  if (_fitModel == PLAN_WAVE || _fitModel == SPHERE_WAVE || _fitModel == ADF)
  {
    chi2 = this->_scan_TDoA();
  }

  if ( _computeErrors )
  {
    if ( ( _fitModel == PLAN_WAVE) && ( _sigma_t <= 0.0 ) )
      this->errorPropagation();
    else
      this->errorPropagation( chi2 );
  }

  return( chi2 );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_scan_TDoA()
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Scan Algorithm for FitTools algorithms. Tries several fit with various
//  initial conditions. For example for point source localisation initial
//  conditions are taken as a scan of the space in spherical coordinates:
//  r, theta, phi. Steps by 15 degrees in theta, half-quadrant (45 degrees)
//  in phi and logarithmically in r, from 1 m to 10 km.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define N_SPEED  1
#define N_THETA  6
#define N_PHI    4
#define N_r      4
#define N_s      4
#define Ncer_width 3
#define Namp_corr  3

{

  //Scan values
  //===
  double speed_v[] = {  0.20, 0.40, 0.60, 0.80, 1.0  };

  //Debug
  // double theta_v[] = {92.92};
  // double phi_v[] = {180.};
  // double r_v[] = {249870.0};
  // double ts_v[] = {88239.0};
  // double cer_width_v[] = {1.37};
  // double amp_corr_v[] = {9.0e7};

  double theta_v[N_THETA];
  for (int i=0; i< N_THETA; i++){
    theta_v[i] = (_xPORT.B( 1, 1 ) + (_xPORT.B( 2, 1 ) - _xPORT.B( 1, 1 ))/(N_THETA-1)*i)*180./M_PI;
    //std::cout<<"theta_v[i] = "<<theta_v[i]<<std::endl;
  }

  double phi_v[N_PHI];
  for (int i=0; i< N_PHI; i++){
    phi_v[i] = (_xPORT.B( 1, 2 ) + (_xPORT.B( 2, 2 ) - _xPORT.B( 1, 2 ))/(N_PHI-1)*i)*180./M_PI;
    //std::cout<<"phi_v[i] = "<<phi_v[i]<<std::endl;
  }

  double r_v[N_r];
  for (int i=0; i< N_r; i++){
    r_v[i] = (_xPORT.B( 1, 3 ) + (_xPORT.B( 2, 3 ) - _xPORT.B( 1, 3 ))/(N_r-1)*i);
    //std::cout<<"r_v[i] = "<<r_v[i]<<std::endl;
  }
  double ts_v[N_s];
  for (int i=0; i< N_s; i++){
    ts_v[i] = (_xPORT.B( 1, 4 ) + (_xPORT.B( 2, 4 ) - _xPORT.B( 1, 4 ))/(N_s-1)*i);
    //std::cout<<"ts_v[i] = "<<ts_v[i]<<std::endl;
  }

  double cer_width_v[Ncer_width];
  for (int i=0; i< Ncer_width; i++){
    cer_width_v[i] = (_xPORT.B( 1, 3 ) + (_xPORT.B( 2, 3 ) - _xPORT.B( 1, 3 ))/(Ncer_width-1)*i);
    //std::cout<<"cer_width_v[i] = "<<cer_width_v[i]<<std::endl;
  }

  double amp_corr_v[Namp_corr];
  for (int i=0; i< Namp_corr; i++){
    amp_corr_v[i] = (_xPORT.B( 1, 4 ) + (_xPORT.B( 2, 4 ) - _xPORT.B( 1, 4 ))/(Namp_corr-1)*i);
    //std::cout<<"amp_corr_v[i] = "<<amp_corr_v[i]<<std::endl;
  }



  double deg2rad = M_PI/180.;

  // Tune scan parameters according to wave type fit
  //===
  int ic_max;

  if ( _fixedSpeed )
    ic_max = 1;
  else
    ic_max = N_SPEED;

  // Do the scan
  //==
  double chi2 = -1;
  double t0;

  std::vector<double> S( _nParameters, 0 );
  for ( int ic = 0; ic < ic_max; ic++ )
    for ( int it = 0; it < N_THETA; it++ )
      for ( int ip = 0; ip < N_PHI; ip++ )
        for (int ir =0; ir < N_r; ir++ )
          for (int is =0; is < N_s; is++ )
            for (int icer_width = 0; icer_width < Ncer_width; icer_width++ )
              for (int iamp_corr = 0; iamp_corr < Namp_corr; iamp_corr++ )
  {
    if ( !_fixedSpeed )
      this->cr() = speed_v[ ic ];

    if ( _fitModel == PLAN_WAVE )
    {
      this->theta() = theta_v[ it ]*deg2rad;
      this->phi()   = phi_v[ ip ]*deg2rad;
    }
    else if ( _fitModel == SPHERE_WAVE )
    {
      this->theta() = theta_v[ it ]*deg2rad;
      this->phi()   = phi_v[ ip ]*deg2rad;
      this->r_xmax()   = r_v[ ir ];
      this->ts()   = ts_v[ is ];
    }

    else if ( _fitModel == ADF )
    {
      this->theta() = theta_v[ it ]*deg2rad;
      this->phi()   = phi_v[ ip ]*deg2rad;
      this->cerenkov_width()   = cer_width_v[ icer_width ];
      this->amp_corr()   = amp_corr_v[ iamp_corr ];
    }

    double d = this->fit();
    if ( ( d >= 0 && d < chi2 ) || ( chi2 < 0 ) )
    {
      chi2   = d;
      for ( int ipar = 0; ipar < _nParameters; ipar++ )
        S[ ipar ] = _xPORT.X( ipar+1 );
    }
  }

  // Restore best parameter guess
  //===
  for ( int ipar = 0; ipar < _nParameters; ipar++ )
    _xPORT.X( ipar+1 ) = S[ ipar ];
  this->_copyFitParameters();

  return chi2;
}
#undef N_PHI
#undef N_THETA
#undef N_SPEED
#undef N_r
#undef Ncer_width
#undef Namp_corr

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::errorPropagation( double chi2obs )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Compute LO error propagation. Build up the error matrix relating input
//  errors on times, DT to output errors on parameters, DT, as:
//
//  DP_i = error( i, j )*DT_j.
//
//  For identical and independently distributed time errors of standard
//  deviation sigma_t(), the standard deviation of the error on parameter
//  P_i goes as:
//
//  sigma_i/sigma_t = error( i ) = sqrt( sum( error( i, j )^2 ), j=1...Na ).
//
//  If a chi2obs value is provided, the routine additionaly computes the
//  significance of the observed chi2 value, assuming identical independently
//  Gausian distributed time errors of standard deviation sigma_t(). The
//  computation is done according to the LO propagation of the error terms.
//
//  NB: Note that the routine is yet only implemented for the plan wave
//  reconstruction algorithm.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( _fitModel != PLAN_WAVE )
    return;

  double A[ 2 ][ 2 ], B[ 2], dK[ 2 ][ 3 ], d1, d2, dx, dy, dz;
  int i, j;

  // Clear error terms
  //===
  memset( _errorM,  0x0, MAX_PARAMETERS*MAX_ANTENNA*sizeof( double ) );
  memset( _errorII, 0x0, MAX_PARAMETERS*sizeof( double ) );

  double ct = cos( _theta );
  double st = sin( _theta );
  double cp = cos(   _phi );
  double sp = sin(   _phi );

  dK[ 0 ][ 0 ] = ct*cp;
  dK[ 0 ][ 1 ] =  ct*sp;
  dK[ 0 ][ 2 ] =    -st;

  dK[ 1 ][ 0 ] = -st*sp;
  dK[ 1 ][ 1 ] = st*cp;
  dK[ 1 ][ 2 ] =    0.0;

  // Compute A matrix ( AX = B, with X = parameters error vector
  // and B = measurements error vector )
  //===
  memset( A, 0x0, 4*sizeof( double ) );
  for ( j = 0; j < _Na-1; j++ )
    for ( i = j+1; i < _Na; i++ )
  {
    dx = _Xa[ j*N_ANTENNA_DATA     ] - _Xa[ i*N_ANTENNA_DATA     ];
    dy = _Xa[ j*N_ANTENNA_DATA + 1 ] - _Xa[ i*N_ANTENNA_DATA + 1 ];
    dz = _Xa[ j*N_ANTENNA_DATA + 2 ] - _Xa[ i*N_ANTENNA_DATA + 2 ];

    d1 = dx*dK[ 0 ][ 0 ] + dy*dK[ 0 ][ 1 ] + dz*dK[ 0 ][ 2 ];
    d2 = dx*dK[ 1 ][ 0 ] + dy*dK[ 1 ][ 1 ];

    A[ 0 ][ 0 ] += d1*d1;
    A[ 1 ][ 1 ] += d2*d2;

    A[ 1 ][ 0 ] += d1*d2;
  }
  A[ 0 ][ 1 ] = A[ 1 ][ 0 ];

  // Compute error matrix terms
  //===
  d1 = A[ 0 ][ 0 ]*A[ 1 ][ 1 ] - A[ 0 ][ 1 ]*A[ 1 ][ 0 ];
  if ( d1 == 0.0 )
    return;
  d1 = 1.0/d1;

  for ( j = 0; j < _Na; j++ )
  {
    memset( B, 0x0, 2*sizeof( double ) );
    for ( i = 0; i < _Na; i++ )
    {
      dx = _Xa[ j*N_ANTENNA_DATA     ] - _Xa[ i*N_ANTENNA_DATA     ];
      dy = _Xa[ j*N_ANTENNA_DATA + 1 ] - _Xa[ i*N_ANTENNA_DATA + 1 ];
      dz = _Xa[ j*N_ANTENNA_DATA + 2 ] - _Xa[ i*N_ANTENNA_DATA + 2 ];
      B[ 0 ] += dx*dK[ 0 ][ 0 ] + dy*dK[ 0 ][ 1 ] + dz*dK[ 0 ][ 2 ];
      B[ 1 ] += dx*dK[ 1 ][ 0 ] + dy*dK[ 1 ][ 1 ];
    }

    _errorM[ 0 ][ j ] = ( B[ 0 ]*A[ 1 ][ 1 ] - B[ 1 ]*A[ 0 ][ 1 ] )*d1;
    _errorM[ 1 ][ j ] = ( B[ 1 ]*A[ 0 ][ 0 ] - B[ 0 ]*A[ 1 ][ 0 ] )*d1;

    _errorII[ 0 ] += _errorM[ 0 ][ j ]*_errorM[ 0 ][ j ];
    _errorII[ 1 ] += _errorM[ 1 ][ j ]*_errorM[ 1 ][ j ];
  }
  _errorII[ 0 ] = sqrt( _errorII[ 0 ] );
  _errorII[ 1 ] = sqrt( _errorII[ 1 ] );


  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Chi2 significance
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  //  Following a LO expansion of the error terms the chi2 writes:
  //
  //  chi2 = DT^t*C^Dt
  //
  //  where C is a symmetric positive define matrix of correlation terms.
  //  We first diagonalise C over an orthonormal basis such that:
  //
  //  chi2 = sum( lambda_i*u_i^2, i=1...Na )
  //
  //  where the u_i's are centred normal distributed. Then we use a
  //  generalised chi2 algorithm to compute the cdf corresponding to the
  //  observation.
  //
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if ( chi2obs == -1.0 )
  {
    _chi2_significance = -1.0;
    return;
  }

  // Build the generalised chi2 correlation matrix
  //===
  double xs[ 3 ], S[ 2 ][ MAX_ANTENNA ];
  double **C = new double*[ _Na ];
  for ( i = 0; i < _Na; i++ )
    C[ i ] = new double[ _Na ];

  memset( xs, 0x0, 3*sizeof( double ) );
  for ( i  = 0; i < _Na; i++ )
  {
    xs[ 0 ]+= _Xa[ i*N_ANTENNA_DATA     ];
    xs[ 1 ]+= _Xa[ i*N_ANTENNA_DATA + 1 ];
    xs[ 2 ]+= _Xa[ i*N_ANTENNA_DATA + 2 ];
  }

  for ( i  = 0; i < _Na; i++ )
  {
    S[ 0 ][ i ] =
      ( xs[ 0 ] - _Na*_Xa[ i*N_ANTENNA_DATA     ] )*dK[ 0 ][ 0 ] +
      ( xs[ 1 ] - _Na*_Xa[ i*N_ANTENNA_DATA + 1 ] )*dK[ 0 ][ 1 ] +
      ( xs[ 2 ] - _Na*_Xa[ i*N_ANTENNA_DATA + 2 ] )*dK[ 0 ][ 2 ];
    S[ 1 ][ i ] =
      ( xs[ 0 ] - _Na*_Xa[ i*N_ANTENNA_DATA     ] )*dK[ 1 ][ 0 ] +
      ( xs[ 1 ] - _Na*_Xa[ i*N_ANTENNA_DATA + 1 ] )*dK[ 1 ][ 1 ];
  }

  for ( j  = 0; j < _Na-1; j++ )
    for ( i  = j+1; i < _Na; i++ )
  {
    C[ i ][ j ] = C[ j ][ i ] =
      _errorM[ 0 ][ i ]*_errorM[ 0 ][ j ]*A[ 0 ][ 0 ] +
      _errorM[ 1 ][ i ]*_errorM[ 1 ][ j ]*A[ 1 ][ 1 ] +
      ( _errorM[ 0 ][ i ]*_errorM[ 1 ][ j ] +
        _errorM[ 1 ][ i ]*_errorM[ 0 ][ j ] )*A[ 0 ][ 1 ] +
      _errorM[ 0 ][ i ]*S[ 0 ][ j ] + _errorM[ 0 ][ j ]*S[ 0 ][ i ] +
      _errorM[ 1 ][ i ]*S[ 1 ][ j ] + _errorM[ 1 ][ j ]*S[ 1 ][ i ] +
      -1.0;
  }

  for ( i  = 0; i < _Na; i++ )
  {
    C[ i ][ i ] =
      _errorM[ 0 ][ i ]*_errorM[ 0 ][ i ]*A[ 0 ][ 0 ] +
      _errorM[ 1 ][ i ]*_errorM[ 1 ][ i ]*A[ 1 ][ 1 ] +
      2.0*_errorM[ 0 ][ i ]*_errorM[ 1 ][ i ]*A[ 0 ][ 1 ] +
      2.0*_errorM[ 0 ][ i ]*S[ 0 ][ i ] +
      2.0*_errorM[ 1 ][ i ]*S[ 1 ][ i ] +
      _Na - 1.0;
  }

  // Compute the generalised chi2 eigen values
  //===
  this->eigenv( C, _lambda, _Na );

  // Protect eigen values and chi2 against roundof errors
  //===
  double lmax, sum;

  lmax = _lambda[ 0 ];
  sum  = 0.0;
  for ( i  = 0; i < _Na; i++ )
  {
    if ( _lambda[ i ] > lmax )
      lmax = _lambda[ i ];
    sum+= _lambda[ i ];
  }

  for ( i  = 0; i < _Na; i++ )
    if ( _lambda[ i ]/lmax < 1e-3 )
      _lambda[ i ] = 0.0;
  if ( chi2obs/sum < 1e-3 )
    chi2obs = 0.0;

  // Build the generalised chi2 coefficients and ndofs
  //===
  int n = 0;
  for ( i = 0; i <_Na; i++ )
  {
    if ( _lambda[ i ] == 0.0 )
     continue;

    _lambda[ n ] = _lambda[ i ];
    _mult[ n ]   = 1;

    for ( j = i+1; j < _Na; j++ )
      if ( fabs( _lambda[ n ] - _lambda[ j ] )/fabs( _lambda[ n ] + _lambda[ j ] ) < 1e-3 )
      {
        _lambda[ j ] = 0.0;
	_mult[ n ]++;
      }
    n++;
  }
  memset( _delta, 0x0, _Na*sizeof( double ) );

  // Compute the chi2 significance
  //===
  _chi2_significance = erfinv( gchi2cdf( _lambda, _delta, _mult, n, chi2obs ) )*sqrt( 2 );

  for ( i = 0; i < _Na; i++ )
    delete[] C[ i ];
  delete[] C;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::randn()
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Gaussian normal centred random variable.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( _randn_parity )
  {
    _randn_parity = false;
    return ( _randn_d2 );
  }
  else
    _randn_parity = true;

  do
  {
    _randn_d1 = 2*drand48() - 1;
    _randn_d2 = 2*drand48() - 1;
    _randn_sq = _randn_d1*_randn_d1 + _randn_d2*_randn_d2;
  }
  while ( _randn_sq > 1. || _randn_sq == 0 );

  _randn_sq = sqrt( -2*log( _randn_sq )/_randn_sq );

  _randn_d1*= _randn_sq;
  _randn_d2*= _randn_sq;

  return ( _randn_d1 );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::erfinv( double p )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Inverse of the error function. Adaptation from an algorithm giving the
//  lower tail quantile standard normal distribution function.
//
//  The original algorithm uses a minimax approximation by rational functions
//  and the result has a relative error whose absolute value is less than
//  1.15e-9.
//
//  Ref: Peter J. Acklam,  http://www.math.uio.no/~jacklam
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  // Coefficients in rational approximations.
  static const double a[] =
  {
    -3.969683028665376e+01,
     2.209460984245205e+02,
    -2.759285104469687e+02,
     1.383577518672690e+02,
    -3.066479806614716e+01,
     2.506628277459239e+00
  };

  static const double b[] =
  {
    -5.447609879822406e+01,
     1.615858368580409e+02,
    -1.556989798598866e+02,
     6.680131188771972e+01,
    -1.328068155288572e+01
  };

  static const double c[] =
  {
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
     4.374664141464968e+00,
     2.938163982698783e+00
  };

  static const double d[] =
  {
     7.784695709041462e-03,
     3.224671290700398e-01,
     2.445134137142996e+00,
     3.754408661907416e+00
  };

  double q, r;

  if ( p <= 0 )
    return ( 0.0 );
  else if ( p >= 1)
    return ( HUGE_VAL );

  p = 0.5*( 1 + p );

  if ( p > 0.97575 ) // Rational approximation for upper region
  {
    q  = sqrt(-2*log(1-p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
	    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1) /
	    sqrt(2);
  }
  else // Rational approximation for central region
  {
    q = p - 0.5;
    r = q*q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
	   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1) /
	   sqrt(2);
  }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::eigenv( double** A, double* V, const int& n )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Compute the eigen values V of the n by n square matrix A. The eigen
//  vectors are overwriten to matrix A.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  static double work[ MAX_ANTENNA ];

  _tred2( A, n, V, work );
  _tqli( V, work, n, A );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::_tred2( double **a, const int& n, double* d, double* e )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Householder reduction of a double, symmetric matrix a[0..n-1][0..n-1]. On
//  output, a is replaced by the orthogonal matrix Q effecting the
//  transformation. d[0..n-1] returns the diagonal elements of the tridiagonal
//  matrix, and e[0..n-1] the off-diagonal elements, with e[0]=0. Several
//  statements, as noted in comments, can be omitted if only eigenvalues are
//  to be found, in which case a contains no useful information on output.
//  Otherwise they are to be included.
//
//  Ref: Numerical Recipes in C, 11.2.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  int l, k, j, i;
  double scale, hh, h, g, f;

  for ( i = n-1; i >= 1; i-- )
  {
    l = i-1;
    h = scale = 0.0;
    if ( l > 0 )
    {
      for ( k = 0; k <= l; k++ )
        scale += fabs( a[ i ][ k]  );
      if ( scale == 0.0 ) // Skip transformation.
        e[ i ] = a [ i ][ l ];
      else
      {
        for ( k = 0; k <= l; k++ )
	{
          a[ i ][ k ] /= scale; // Use scaled a's for transformation.
          h += a[ i ][ k ]*a[ i ][ k ]; // Form sigma in h.
        }
        f = a[ i ][ l ];
        g = ( f >= 0.0 ? -sqrt( h ) : sqrt( h ) );
        e[ i ] = scale*g;
        h -= f*g; // Now h is equation (11.2.4).
        a[ i ][ l ] = f - g; //Store u in the ith row of a.
        f = 0.0;
        for ( j = 0; j <= l; j++ )
	{
          /* Next statement can be omitted if eigenvectors not wanted */
          a[ j ][ i ] = a[ i ][ j ]/h; // Store u/H in ith column of a.
          g = 0.0; // Form an element of A.u in g.
          for ( k = 0; k <= j; k++ )
            g += a[ j ][ k ]*a[ i ][ k ];
          for ( k = j+1; k <= l; k++ )
            g += a[ k ][ j ]*a[ i ][ k ];
          e[ j ] = g/h; // Form element of p in temporarily unused element of e.
	  f += e[ j ]*a[ i ][ j ];
        }
        hh = f/( h + h ); // Form K, equation (11.2.11).
        for ( j = 0; j <= l; j++ ) // Form q and store in e overwriting p.
	{
          f = a[ i ][ j ];
          e[ j ] = g = e[ j ] - hh*f;
          for ( k = 0; k <= j; k++ ) // Reduce a, equation (11.2.13).
          a[ j ][ k ] -= ( f*e[ k ] + g*a[ i ][ k ] );
        }
      }
    }
    else
      e[ i ] = a[ i ][ l ];
    d[ i ] = h;
  }
  /* Next statement can be omitted if eigenvectors not wanted */
  d[ 0 ] = 0.0;
  e[ 0 ] = 0.0;
  /* Contents of this loop can be omitted if eigenvectors not
     wanted except for statement d[ i ] = a [ i ][ i ]; */
  for ( i = 0; i < n; i++ ) // Begin accumulation of transformation matrices.
  {
    l = i - 1;
    if ( d[ i ] ) // This block skipped when i=0
    {
      for ( j = 0; j <= l; j++ )
      {
        g = 0.0;
        for ( k = 0; k <= l; k++ ) // Use u and u/H stored in a to form P.Q.
          g += a[ i ][ k ]*a[ k ][ j ];
        for ( k = 0; k <= l; k++ )
          a[ k ][ j ] -= g*a[ k ][ i ];
      }
    }
    d[ i ] = a[ i ][ i ]; // This statement remains.
    a[ i ][ i ] = 1.0;    // Reset row and column of a to identity
    for ( j = 0; j <= l; j++ ) // matrix for next iteration.
      a[ j ][ i ] = a[ i ][ j ] = 0.0;
  }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::_tqli( double* d, double* e, const int& n, double **z )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  QL algorithm with implicit shifts, to determine the eigenvalues and
//  eigenvectors of a double, symmetric, tridiagonal matrix, or of a double,
//  symmetric matrix previously reduced by tred2 (11.2). On input, d[0..n-1]
//  contains the diagonal elements of the tridiagonal matrix. On output, it
//  returns the eigenvalues. The vector e[0..n-1] inputs the subdiagonal
//  elements of the tridiagonal matrix, with e[0] arbitrary. On output e
//  is destroyed. When finding only the eigenvalues, several lines may be
//  omitted, as noted in the comments. If the eigenvectors of a tridiagonal
//  matrix are desired, the matrix z[0..n-1][0..n-1] is input as the identity
//  matrix. If the eigenvectors of a matrix that has been reduced by tred2
//  are required, then z is input as the matrix output by tred2. In either
//  case, the kth column of z returns the normalized eigenvector
//  corresponding to d[k].
//
//  Ref: Numerical Recipes in C, 11.3.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define SIGN( a, b ) ( (b) >= 0.0 ? fabs(a) : -fabs(a) )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  int m, l, iter, i, k;
  double s, r, p, g, f, dd, c, b;

  for ( i = 1; i < n; i++ ) // Convenient to renumber the elements of e.
    e[ i-1 ] = e[ i ];
  e[ n-1 ] = 0.0;

  for ( l = 0; l < n; l++ )
  {
    iter=0;
    do
    {
      for ( m = l; m < n-1; m++ ) // Look for a single small subdiagonal element to split the matrix.
      {
        dd = fabs( d[ m ] ) + fabs( d[ m+1 ] );
        if ( (double)( fabs( e[ m ] ) + dd ) == dd )
	  break;
      }
      if ( m != l )
      {
        if ( iter++ == 30 )
	{
	  printf( "Too many iterations in tqli\n" );
	  return;
	}
        g = ( d[ l+1 ] - d[ l ] )/( 2.0*e[ l ] ); // Form shift.
        r = _pythag( g, 1.0 );
        g = d[ m ] - d[ l ] + e[ l ]/( g + SIGN( r, g ) ); // This is dm - ks.
        s = c = 1.0;
        p = 0.0;
        for ( i = m-1; i >= l; i-- ) // A plane rotation as in the original QL, followed by Givens
	{                            // rotations to restore tridiagonal form.
          f = s*e[ i ];
          b = c*e[ i ];
          e[ i+1 ] = ( r = _pythag( f, g ) );
          if ( r == 0.0 ) // Recover from underflow.
	  {
	    d[ i+1 ] -= p;
            e[ m ] = 0.0;
            break;
          }
          s = f/r;
          c = g/r;
          g = d[ i+1 ] - p;
          r = ( d[ i ] - g )*s + 2.0*c*b;
          d[ i+1 ] = g + ( p = s*r );
          g = c*r - b;
          /* Next loop can be omitted if eigenvectors not wanted*/
          for ( k = 0; k < n; k++ ) // Form eigenvectors.
	  {
            f = z[ k ][ i+1 ];
            z[ k ][ i+1 ] = s*z[ k ][ i ] + c*f;
            z[ k ][ i ]   = c*z[ k ][ i ] - s*f;
          }
        }
        if ( r == 0.0 && i >= l )
	  continue;
        d[ l ] -= p;
        e[ l ]  = g;
        e[ m ] = 0.0;
      }
    }
    while ( m != l );
  }
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#undef SIGN
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_pythag ( const double& a, const double& b )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  sqrt(a*a+b*b) with protection against under/overflow
//
//  Ref: Numerical Recipes in C, 2.6.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
   double wa, wb, w;

   wa = fabs ( a );
   wb = fabs ( b );

   if ( wa > wb )
   {
      w = wa;
      wa = wb;
      wb = w;
   }

   if ( wb == 0.0 )
      return 0.0;
   else
   {
      w = wa / wb;
      return wb * sqrt ( 1.0 + w * w );
   }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::gchi2cdf( double* lambda, double* delta, int* ndof,
  const int& n, const double& x )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Computes the cdf of the generalised chi2 distribution:
//
//  gchi2 = sum[ lambda_i*nc_chi2( delta_i, ndof_i ), i=0,n-1 ]
//
//  where nc_chi2( delta, ndof ) is a non centred chi2 distributed variable
//  with ndof degrees of freedom and non-centrality parameter delta.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  static const int	order_153	= 12;
  static const double	eps1_256	= 1e-4;
  static const double	eps2_256	= 1e-4;
  static const double	eps3_256	= 1e-4;

  static const double	eps_204		= 1e-4;
  static const double	eps_155		= 1e-4;
  static const int      lim_155         = 10000;

  double 		bound_256  	= 0.0;
  int    		ifault   	= 0;

  if ( ( n == 1 ) && ( _mult[ 0 ] == 1 ) )
    return( erf( x/sqrt( 2 )/lambda[ 0 ] ) );

  if ( _gchi2_algorithm == 256 )
    return ( _imhof( lambda, ndof, delta, n, x, bound_256,
      eps1_256, eps2_256, eps3_256, ifault ) );
  else if ( _gchi2_algorithm == 204 )
    return( _ruben( lambda, ndof, delta, n, x, eps_204, ifault ) );
  else if ( _gchi2_algorithm == 155 )
    return( _qf( lambda, delta, ndof, n, 0.0, x, lim_155, eps_155, _trace_155, &ifault ) );
  else if ( _gchi2_algorithm == 153 )
  {
    memcpy( _p_lambda_153, lambda, n*sizeof( double ) ); // lambda[ 0 ] = 0.0 is used to pass
                                                         // an additional parameter not relevant here. See AS153.f for details.
							 // In addition lambda values should be ordered and differents for this algorithm
							 // to work.
    return( gradsol_( _lambda_153, n, x, order_153 ) );
  }
  else
    return ( -1.0 );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_imhof( double* lambda, int* mult, double* delta,
  const int& noterms, const double& arg, double& bound,
  const double& eps1, const double& eps2, const double& eps3, int& ifault )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  C++ translation of Koerts and Abrahamse's implementation of
//  Imhof's procedure for evaluating the probability that a diagonal
//  form in normal variables is less than a given value, arg.
//
//  Ref: Algorithm AS 256.3  Appl. Statist. (1990) Vol.39, No.2.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define SECURED_IMHOF 0
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  #if ( SECURED_IMHOF )
    // Check parameters
    //===
    if ( ( noterms < 1 ) || ( eps1 <= 0.0 ) ||
         ( eps2 <= 0.0 ) || ( eps3 <= 0.0 ) )
    {
      ifault = 2;
      return( -2.0 );
    }

    ifault = 0;
    for ( int i = 0; i < noterms; i++ )
    {
      if ( ( mult[ i ] < 1 ) || ( delta[ i ] < 0.0 ) )
      {
        ifault = 3;
        return( -(double)i );
      }
    }
  #endif

  // Main body
  //===
  if ( bound <= 0.0 )
    bound = _imhofbd( lambda, mult, delta, noterms, eps1, eps2, ifault );

  double p = _imhofint( lambda, mult, delta, noterms, arg, bound, eps3, ifault );
  if ( ( p < 1e-4 ) || ( ifault != 0 )  )
    p = 0.0;

  return( p );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_imhofbd( double* lambda, int* mult, double* delta,
  const int& noterms, const double& eps1, const double& eps2, int& ifault )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  C++ translation of Koerts and Abrahamse's procedure
//  for evaluating Imhof's upper bound.
//
//  Ref: Algorithm AS 256.1  Appl. Statist. (1990) Vol.39, No.2.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  int i;
  double count, hold, range, sum1, sum2;

  #if ( SECURED_IMHOF )
    if ( ( noterms < 1 ) || ( eps1 <= 0.0 ) || ( eps2 <= 0.0 ) )
    {
      ifault = 2;
      return( -2.0 );
    }
  #endif

  count = sum1 = sum2 = 0.0;
  for( i = 0; i < noterms; i++ )
  {
    hold = fabs( lambda[ i ] );
    if ( hold > eps2 )
    {
      count += (double)mult[ i ];
      sum1  += mult[ i ]*log( hold );
    }
    sum2 += delta[ i ];
  }

  if ( count < 0.9 )
  {
    ifault = 4;
    return( -4.0 );
  }

  count *= 0.5;
  sum1   = 0.5*sum1 + log( M_PI*count );
  range  = exp( -( sum1 + 0.5*sum2 + log( eps1 ) )/count );

  if ( sum2 == 0.0 )
    range += 5.0/count;
  else
  {
    do
    {
      sum2 = 0.0;
      for ( i = 0; i < noterms; i++ )
      {
        hold  = range*lambda[ i ];
	hold *= hold;
        sum2 += delta[ i ]*hold/( 1.0 + hold );
      }
      hold = exp( sum1 + count*log( range ) + 0.5*sum2 );
      if ( hold*eps1 <= 1.0 )
        range += 5.0/count;
      else
        count = 0.0;
    }
    while ( count != 0.0 );
  }

  return( range );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_imhofint( double* lambda, int* mult, double* delta,
  const int& noterms, const double& arg, const double& bound,
  const double& eps3, int& ifault )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  C++ translation of Koerts and Abrahamse's procedure
//  for evaluating Imhof's integral by Simpson's Rule.
//
//  Ref: Algorithm AS 256.2  Appl. Statist. (1990) Vol.39, No.2.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  static const int maxit = 14;

  int i, j ,n;
  double eps4, int1, int2, step, sum1, sum2, sum4;
  bool cgd;

  #if( SECURED_IMHOF )
    if ( ( noterms < 1 ) || ( bound <= 0.0 ) || ( eps3 <= 0.0 ) )
    {
      ifault = 2;
      return( -2.0 );
    }
  #endif

  ifault = 5;
  n      = 2;
  step   = 0.5*bound;
  eps4   = 3.0*M_PI*eps3;

  sum1   = -arg;
  for ( i = 0; i < noterms; i++ )
    sum1 +=lambda[ i ]*( mult[ i ] + delta[ i ] );
  sum1 = 0.5*sum1 + _imhoffn( lambda, mult, delta, noterms, arg, bound );

  sum4 = _imhoffn( lambda, mult, delta, noterms, arg, step );
  int2 = ( sum1 + 4.0*sum4 )*step;

  sum2 = 0.0;
  for ( i = 1; i <= maxit; i++ )
  {
    n+= n;
    step  = 0.5*step;

    sum2 += sum4;
    sum4  = 0.0;
    for ( j = 1; j <= n; j+= 2 )
      sum4 += _imhoffn( lambda, mult, delta, noterms, arg, j*step );

    int1 = int2;
    int2 = ( sum1 + 2.0*sum2 + 4.0*sum4 )*step;

    if ( i > 3 )
      if ( fabs( int1 - int2 ) < eps4 )
      {
        if ( fabs( int2 ) > 1.5*M_PI )
	  ifault = 6;
	else
	  ifault = 0;
        break;
      }
  }

  return( 0.5 - int2/( 3.0*M_PI ) );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_imhoffn( double* lambda, int* mult, double* delta,
  const int& noterms, const double& arg, const double& u )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// imhoffn evaluates Imhof's integrand.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  int i;
  double hold, hold2, hold3, rho, sum, theta;

  rho   = 0.0;
  sum   = 0.0;
  theta = -u*arg;

  for ( i  = 0; i < noterms; i++ )
  {
    hold   = u*lambda[ i ];
    hold2  = hold*hold;
    hold3  = 1.0 + hold2;
    theta += mult[ i ]*atan( hold ) + delta[ i ]*hold/hold3;
    sum   += delta[ i]*hold2/hold3;
    rho   += mult[ i ]*log( hold3 );
  }

  return( sin( 0.5*theta )/( u*exp( 0.5*sum + 0.25*rho ) ) );
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#undef SECURED_IMHOF
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_ruben( double* lambda, int* mult, double* delta,
  const int& n, const double& c, const double& eps, int& ifault )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Use of Rubin's (1962) method to evaluate the expression
//  Pr[d k(i) K}(m(i),k(i)}) < c]   where k(i) and c are positive constants,
//  and where K}(m(i),k(i)}) represents an independent chi-squared random
//  variable with m(j) degrees of freedom and non-centrality parameter k(i)}.
//
//   n =  number of chi-squared terms
//   c =  Critical chi-squared value
//   maxit = maximum number of iterations = 500 in Vol 33 No. 3 1984
//   eps = degree of accuracy
//
//  The program returns the cumulative probability value at the point c in
//  the distribution.
//
//  Ref: Program description: Journal of the Royal Statistical Society
//  (Series C) Vol 33 No 3  1984, R.W. Farebrother,  Algorithm AS204,
//  pp 332 - 339.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  static const int    maxit  = 500;                  // Maximum number of iterations in eq. 3 of algorithm.
  static const double mode   = 0.0;                  // Set mode = 0.90625 for AS 204A. But 0.0 should be
                                                     // closer to optimal beta choice (see Rubin's).
  static const double lnspi2 = 0.5*log( 0.5*M_PI );

  double tol, beta, ao, aoinv, dnsty, z, rz, eps2, hold, hold2, sum, sum1, dans,
    lans, pans, prbty, temp;
  int itemp, i, j, k, m;
  double gamma[ MAX_ANTENNA ], theta[ MAX_ANTENNA ], a[ maxit ],
    b[ maxit ];
  bool ext, L;

  if ( ( n < 1 ) || ( c <= 0.0 ) || ( maxit < 1 ) || ( eps <= 0.0 ) )
  {
    ifault = 2;
    return( 0.0 );
  }

  tol = -200.0;
  // preliminaries
  beta = sum = lambda[ 0 ];
  for ( i = 0; i < n; i++ )
  {
     if ( ( lambda[ i  ] <= 0.0 ) || ( mult[ i ] < 1 ) || ( delta[ i ] < 0.0 ) )
     {
        ifault = -i;
        return( 0.0 );
     }
     if ( beta > lambda[ i  ] )
       beta = lambda[ i  ];
     if ( sum < lambda[ i ] )
       sum = lambda[ i ];
  }

  if ( mode > 0.0 )
    beta *= mode;
  else
    beta = 2.0/( 1.0/beta + 1.0/sum );

  k    = 0;
  sum  = 1.0;
  sum1 = 0.0;
  for ( i = 0; i < n; i++ )
  {
    hold = beta/lambda[ i ];
    gamma[ i ] = 1.0 - hold;
    for ( j = 0; j < mult[ i ]; j++ )
      sum *= hold;
    sum1 += delta[i];
    k    += mult[ i ];
    theta[ i ] = 1.0;
  }

  ao = exp( 0.5*( log( sum ) - sum1 ) );
  if ( ao <= 0.0 )
  {
    ifault = 1;
    return( 0.0 );
  }

  z = c/beta;
  /* Evaluate probability and density of chi-squared on
     k degrees of freedom. */

  itemp = ( k / 2 ) * 2;
  if (  k == itemp )
  {
    i    = 2;
    lans = -0.5*z;
    dans = exp( lans );
    pans = 1.0 - dans;
  }
  else
  {
    i    = 1;
    lans = -0.5*( z + log( z ) ) - lnspi2;
    dans = exp( lans );
    pans = erf( sqrt( 0.5*z ) );
  }

  k-= 2;

  while( i <= k )
  {
    if ( lans < tol )
    {
      lans += log( z/i );
      dans  = exp( lans );
     }
     else
     {
       temp = dans;
       dans = temp * z/i;
     }

     temp = pans;
     pans = temp - dans;
     i+= 2;
  }

  // Evaluate successive terms of expansion
  prbty = pans;
  dnsty = dans;
  eps2  = eps/ao;
  aoinv = 1.0/ao;
  sum   = aoinv - 1.0;

  for ( m = 0; m < maxit; m++ )
  {
    sum1 = 0.0;
    for ( i = 0; i < n; i++ )
    {
      hold = theta[ i ];
      theta[ i ] *= gamma[ i ];
      sum1 += theta[ i ]*mult[ i ] + m*delta[ i ]*( hold - theta[ i ] );
    }
    sum1 = b[ m ] = 0.5*sum1;
    for ( i = m-1; i >= 0; i-- )
        sum1 += b[ i ]*a[ m - i - 1 ];
    a[ m ] = sum1/( m + 1 );
    sum1 = a[ m ];

    k+= 2;

    if ( lans < tol )
    {
      lans += log( z/k );
      dans  = exp( lans );
    }
    else
    {
      temp = dans;
      dans = temp*z/k;
    }
    pans  -= dans;
    sum   -= sum1;
    dnsty += dans*sum1;
    sum1  *= pans;
    prbty += sum1;

    if ( prbty < -aoinv )
    {
      ifault = 3;
      return ( 0.0 );
    }

    if ( fabs( pans*sum ) < eps2 )
    {
      if ( fabs( sum1 ) < eps2 )
      {
        ifault = 0;
	break;
      }
    }
  }
  if ( m == maxit )
    ifault = 4;

  dnsty *= ao/(beta + beta);
  prbty *= ao;
  if ( ( prbty < 0.0 ) || ( prbty > 1.0 ) )
    ifault += 5;
  else if ( dnsty < 0.0 )
    ifault += 6;

  return( prbty );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_qf( double* lb, double* nc, int* n, const int& r,
  const double& sigma, const double& c, const int& lim, const double& acc,
  double* trace, int* ifault )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Distribution function of a linear combination of non-central
//  chi-squared random variables :
//
//  input:
//  lb[j]            coefficient of j-th chi-squared variable
//  nc[j]            non-centrality parameter
//  n[j]             degrees of freedom
//  j = 0, 2 ... r-1
//  sigma            coefficient of standard normal variable
//  c                point at which df is to be evaluated
//  lim              maximum number of terms in integration
//  acc              maximum error
//
//  output:
//  ifault = 1       required accuracy NOT achieved
//           2       round-off error possibly significant
//           3       invalid parameters
//           4       unable to locate integration parameters
//           5       out of memory
//
// trace[0]         absolute sum
// trace[1]         total number of integration terms
// trace[2]         number of integrations
// trace[3]         integration interval in final integration
// trace[4]         truncation point in initial integration
// trace[5]         s.d. of initial convergence factor
// trace[6]         cycles to locate integration parameters
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define LOG28 .0866 // log( 2.0 ) / 8.0
//-------------------------------------------------------------------------
#define square( X ) \
  (X)*(X)
//-------------------------------------------------------------------------
#define cube( X ) \
  (X)*(X)*(X)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  int j, nj, nt, ntm;
  double acc1, almx, xlim, xnt, xntm;
  double utx, tausq, sd, intv, intv1, x, up, un, d1, d2, lj, ncj;
  double qfval;
  static const int rats[] = { 1, 2, 4, 8 };

  if ( setjmp( _qf_env ) != 0 ) // On call to longjmp, jump to endofproc
  {
    *ifault = 4;
    goto endofproc;
  }

  _qf_r   = r;
  _qf_lim = lim;
  _qf_c   = c;
  _qf_n   = n;
  _qf_lb  = lb;
  _qf_nc  = nc;

  memset( trace, 0x0, 7*sizeof( double ) );

  *ifault    = 0;
  _qf_count  = 0;
  _qf_intl   = 0.0;
  _qf_ersm   = 0.0;
  qfval      = -1.0;
  acc1       = acc;
  _qf_ndtsrt = true;
  _qf_fail   = false;
  xlim       = (double)_qf_lim;

  _qf_th = (int*)malloc( _qf_r*sizeof( int ) );
  if ( !_qf_th )
  {
    *ifault = 5;
    goto endofproc;
  }

  // Find mean, sd, max and min of lb,
  // check that parameter values are valid.
  _qf_sigsq = square( sigma );
  sd        = _qf_sigsq;
  _qf_lmax  = 0.0;
  _qf_lmin  = 0.0;
  _qf_mean  = 0.0;
  for ( j = 0; j < _qf_r; j++ )
  {
    nj  = _qf_n[  j ];
    lj  = _qf_lb[ j ];
    ncj = _qf_nc[ j ];
    if ( nj < 0  ||  ncj < 0.0 )
    {
      *ifault = 3;
      goto  endofproc;
    }
    sd  += square( lj )*( 2.0*nj + 4.0*ncj );
    _qf_mean += lj*( nj + ncj );
    if ( _qf_lmax < lj )
      _qf_lmax = lj ;
    else if ( _qf_lmin > lj )
      _qf_lmin = lj;
  }

  if ( sd == 0.0  )
  {
    qfval = ( _qf_c > 0.0 ) ? 1.0 : 0.0;
    goto  endofproc;
  }
  if ( ( _qf_lmin == 0.0 ) && ( _qf_lmax == 0.0 ) && ( sigma == 0.0 ) )
  {
    *ifault = 3;
    goto  endofproc;
  }
  sd   = sqrt( sd );
  almx = ( _qf_lmax < -_qf_lmin ) ? -_qf_lmin : _qf_lmax;

  // Starting values for findu, ctff
  utx = 16.0/sd;
  up  = 4.5/sd;
  un  = -up;

  // Truncation point with no convergence factor.
  _qf_findu( &utx, 0.5*acc1 );

  // Does convergence factor help?
  if ( _qf_c != 0.0  && ( almx > 0.07 * sd ) )
  {
    tausq = .25 * acc1 / _qf_cfe( _qf_c );
    if ( _qf_fail )
      _qf_fail = false ;
    else if ( _qf_truncation( utx, tausq ) < .2 * acc1 )
    {
      _qf_sigsq += tausq;
      _qf_findu( &utx, .25 * acc1 );
      trace[ 5 ] = sqrt( tausq );
    }
  }
  trace[ 4 ] = utx;
  acc1       = 0.5 * acc1;

  // Find RANGE of distribution, quit if outside this.
  l1:
    d1 = _qf_ctff( acc1, &up ) - _qf_c;
    if ( d1 < 0.0 )
    {
      qfval = 1.0;
      goto endofproc;
    }
    d2 = _qf_c - _qf_ctff( acc1, &un );
    if ( d2 < 0.0 )
    {
      qfval = 0.0;
      goto endofproc;
    }

    // Find integration interval.
    intv = 2.0 * M_PI / ( ( d1 > d2 ) ? d1 : d2 );

    // Calculate number of terms required for main and
    // auxillary integrations.
    xnt  = utx / intv;
    xntm = 3.0 / sqrt( acc1 );
    if ( xnt > xntm * 1.5 )
    {
      // Parameters for auxillary integration
      if ( xntm > xlim )
      {
        *ifault = 1;
	goto endofproc;
      }
      ntm   = (int)floor( xntm + 0.5 );
      intv1 = utx / ntm;
      x     = 2.0 * M_PI / intv1;
      if ( x <= fabs( _qf_c ) )
        goto l2;

      // Calculate convergence factor.
      tausq = 0.33 * acc1 / ( 1.1 * ( _qf_cfe( _qf_c - x ) + _qf_cfe( _qf_c + x ) ) );
      if ( _qf_fail )
        goto l2;
      acc1 = 0.67 * acc1;

      // Auxillary integration.
      _qf_integrate( ntm, intv1, tausq, false );
      xlim  -= xntm;
      _qf_sigsq += tausq;
      trace[ 2 ] += 1.0;
      trace[ 1 ] += ntm + 1.0;

      // Find truncation point with new convergence factor.
      _qf_findu( &utx, .25 * acc1 );
      acc1 = 0.75 * acc1;
      goto l1;
    }

    // Main integration.
    l2:
      trace[ 3 ] = intv;
      if ( xnt > xlim )
      {
        *ifault = 1;
	goto endofproc;
      }
      nt = (int)floor( xnt + 0.5 );
      _qf_integrate( nt, intv, 0.0, true );
      trace[ 2 ] += 1.0;
      trace[ 1 ] += nt + 1.0;
      qfval       = 0.5 - _qf_intl;
      trace[ 0]   = _qf_ersm;

      // Test whether round-off error could be significant
      // allow for radix 8 or 16 machines.
      up = _qf_ersm;
      x  = up + acc / 10.0;
      for ( j = 0; j < 4; j++ )
        if ( rats[ j ] * x == rats[ j ] * up )
	  *ifault = 2;

    endofproc:
      free( (char*)_qf_th );
      trace[ 6 ] = (double)_qf_count;
      return qfval;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::_qf_counter()
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Count number of calls to errbd, truncation, cfe.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  _qf_count++;

  if ( _qf_count > _qf_lim )
    longjmp( _qf_env, 1 );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_qf_exp1( double x )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// To avoid underflows.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  return ( x < -50.0 ? 0.0 : exp( x ) );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_qf_log1( const double& x, const bool& first )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  If ( first ) log( 1 + x ) ; else  log( 1 + x ) - x.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  if ( fabs( x ) > 0.1 )
    return ( first ? log( 1.0 + x ) : ( log( 1.0 + x ) - x ) );
  else
  {
    double s, s1, term, y, k;

    y     = x / ( 2.0 + x );
    term  = 2.0 * cube( y );
    k     = 3.0;
    s     = ( first ? 2.0 : - x )*y;
    y     = square( y );
    for ( s1 = s + term / k; s1 != s; s1 = s + term / k )
    {
      k    += 2.0;
      term *= y;
      s     = s1;
    }
    return s;
  }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::_qf_order()
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Find order of absolute values of lb
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  int j, k;
  double lj;

  for ( j = 0; j < _qf_r; j++ )
  {
    lj = fabs( _qf_lb[ j ] );
    for ( k = j-1; k >= 0; k-- )
    {
      if ( lj > fabs( _qf_lb[ _qf_th[ k ] ] ) )
        _qf_th[ k + 1] = _qf_th[ k ];
      else goto l1;
    }
    k = -1;
    l1 :
      _qf_th[ k + 1 ] = j;
  }
  _qf_ndtsrt = false;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_qf_errbd( double u, double* cx )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Find bound on tail probability using mgf, cutoff
//  point returned to *cx.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  double sum1, lj, ncj, x, y, xconst;
  int j, nj;

  _qf_counter();

  xconst = u * _qf_sigsq;
  sum1 = u * xconst;
  u *= 2.0;

  for ( j = _qf_r-1; j >= 0; j-- )
  {
    nj  = _qf_n[  j ];
    lj  = _qf_lb[ j ];
    ncj = _qf_nc[ j ];
    x   = u * lj;
    y = 1.0 - x;
    xconst += lj * ( ncj / y + nj ) / y;
    sum1   += ncj * square( x / y )
           + nj * ( square( x ) / y + _qf_log1( -x, false ) );
  }
  *cx = xconst;

  return _qf_exp1( -0.5 * sum1 );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_qf_ctff( double accx, double* upn )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Find ctff so that p(qf > ctff) < accx,  if (upn > 0),
//  p(qf < ctff) < accx, otherwise.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  double u1, u2, u, rb, xconst, c1, c2;

  u2 = *upn;
  u1 = 0.0;
  c1 = _qf_mean;
  rb = 2.0 * ( ( u2 > 0.0 ) ? _qf_lmax : _qf_lmin );
  for ( u = u2 / ( 1.0 + u2 * rb ); _qf_errbd( u, &c2 ) > accx; u = u2 / ( 1.0 + u2 * rb ) )
  {
    u1 = u2;
    c1 = c2;
    u2 = 2.0 * u2;
  }
  for ( u = ( c1 - _qf_mean ) / ( c2 - _qf_mean ); u < 0.9; u = ( c1 - _qf_mean ) / ( c2 - _qf_mean ) )
  {
    u = ( u1 + u2 ) / 2.0;
    if ( _qf_errbd( u / (1.0 + u * rb ), &xconst ) > accx )
    {  u1 = u;
       c1 = xconst;
    }
    else
    {
      u2 = u;
      c2 = xconst;
    }
  }
  *upn = u2;

  return c2;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_qf_truncation( double u, double tausq )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Bound integration error due to truncation at u.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  double sum1, sum2, prod1, prod2, prod3, lj, ncj,
    x, y, err1, err2;
  int j, nj, s;

  _qf_counter();

  sum1   = 0.0;
  prod2  = 0.0;
  prod3  = 0.0;
  s      = 0;
  sum2   = ( _qf_sigsq + tausq )*u*u;
  prod1  = 2.0*sum2;
  u     *= 2.0;
  for ( j = 0; j < _qf_r; j++ )
  {
    lj    = _qf_lb[ j ];
    ncj   = _qf_nc[ j ];
    nj    = _qf_n[  j ];
    x     = u*u*lj*lj;
    sum1 += ncj*x/( 1.0 + x );

    if ( x > 1.0 )
    {
      prod2 += nj*log( x );
      prod3 += nj*_qf_log1( x, true );
      s     += nj;
    }
    else
      prod1 += nj*_qf_log1( x, true );
  }
  sum1  *= 0.5;
  prod2 += prod1;
  prod3 += prod1;
  x = _qf_exp1( -sum1 - 0.25*prod2 ) / M_PI;
  y = _qf_exp1( -sum1 - 0.25*prod3 ) / M_PI;
  err1 = ( s  ==  0 )  ? 1.0 : x*2.0/s;
  err2 = ( prod3 > 1.0 ) ? 2.5*y : 1.0;
  if ( err2 < err1 )
    err1 = err2;
  x = 0.5*sum2;
  err2 = ( x  <=  y ) ? 1.0 : y/x;

  return ( err1 < err2 ) ? err1 : err2;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::_qf_findu( double* utx, double accx )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Find u such that truncation( u ) < accx and truncation( u / 1.2 ) > accx.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  double u, ut;
  int i;
  double divis[] = { 2.0, 1.4, 1.2, 1.1 };

  ut = *utx;
  u  = 0.25*ut;
  if ( _qf_truncation( u, 0.0 ) > accx )
  {
    for ( u = ut; _qf_truncation( u, 0.0 ) > accx; u = ut )
      ut *= 4.0;
  }
  else
  {
    ut = u;
    for ( u = 0.25*u; _qf_truncation( u, 0.0 ) <=  accx; u = 0.25*u )
      ut = u;
  }
  for ( i = 0; i < 4; i++ )
  {
    u = ut/divis[ i ];
    if ( _qf_truncation( u, 0.0 )  <=  accx )
      ut = u;
  }
  *utx = ut;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void FitTools::_qf_integrate( int nterm, double interv, double tausq, bool mainx )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Carry out integration with nterm terms, at stepsize
//  interv.  if (! mainx ) multiply integrand by
//  1.0-exp(-0.5*tausq*u^2).
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  double inpi, u, sum1, sum2, sum3, x, y, z;
  int k, j, nj;

  inpi = interv / M_PI;
  for ( k = nterm; k >= 0; k-- )
  {
    u = ( k + 0.5 )*interv;
    sum1 = - 2.0*u*_qf_c;
    sum2 = fabs( sum1 );
    sum3 = -0.5*_qf_sigsq*square( u );
    for ( j = _qf_r-1; j >= 0; j-- )
    {
      nj    = _qf_n[ j ];
      x     = 2.0*_qf_lb[ j ]*u;
      y     = square( x );
      sum3 -= 0.25*nj*_qf_log1( y, true );
      y     = _qf_nc[ j ]* x/( 1.0 + y );
      z     = nj*atan( x ) + y;
      sum1 += z;
      sum2 += fabs( z );
      sum3 -= 0.5*x*y;
    }

    x = inpi*_qf_exp1( sum3 )/u;
    if ( !mainx )
      x *= 1.0 - _qf_exp1( -0.5*tausq*square( u ) );
    sum1      = sin( 0.5*sum1 )*x;
    sum2     *= 0.5*x;
    _qf_intl += sum1;
    _qf_ersm += sum2;
  }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double FitTools::_qf_cfe( double x )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Coef of tausq in error when convergence factor of
// exp1(-0.5*tausq*u^2) is used when df is evaluated at x.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  double axl, axl1, axl2, sxl, sum1, lj;
  int j, k, t;

  _qf_counter();

  if ( _qf_ndtsrt )
    _qf_order();
  axl  = fabs( x );
  sxl  = ( x > 0.0 ) ? 1.0 : -1.0;
  sum1 = 0.0;

  for ( j = _qf_r-1; j >= 0; j-- )
  {
    t = _qf_th[ j ];
    if ( _qf_lb[ t ] * sxl > 0.0 )
    {
      lj   = fabs( _qf_lb[ t ] );
      axl1 = axl - lj*( _qf_n[t] + _qf_nc[ t ] );
      axl2 = lj / LOG28;
      if ( axl1 > axl2 )
        axl = axl1;
      else
      {
        if ( axl > axl2 )
	  axl = axl2;
        sum1 = ( axl - axl1 ) / lj;
        for ( k = j-1; k >= 0; k-- )
          sum1 += ( _qf_n[ _qf_th[ k ] ] + _qf_nc[ _qf_th[ k ] ] );
        goto l;
      }
    }
  }

  l:
    if ( sum1 > 100.0 )
    {
      _qf_fail = true;
      return 1.0;
    }
    else
      return pow( 2.0, ( sum1 / 4.0 ) ) / ( M_PI*square( axl ) );
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#undef LOG28
#undef square
#undef cube
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//Shit load of various functions needed for minimisation functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double GetRefractionIndexAtXmax(double x0,double y0,double z0,double ns, double kr)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//Aires refractive index from Matias Tueros computation
{
    double R02, rh0, h0, n_h0;
    double rearth=6370949.0;
    R02=x0*x0+y0*y0;  //notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada
    h0=(sqrt((z0+rearth)*(z0+rearth) + R02 ) - rearth)/1.E3; //altitude of emission, in km

    rh0 = ns*exp(kr*h0); //refractivity at emission
    n_h0=1.E0+1.E-6*rh0; //n at emission

    return n_h0;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double _EffectiveRefractionIndex(double x0,double y0,double z0,double ns, double kr, double groundz,double xant,double yant,int stepsize)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//Aires refractive index from Matias Tueros computation
{
      double R02, h0, rh0, n_h0, hd, ux, uy, uz, Rd, kx, ky, kz, currpx, currpy, currpz, currh, avn, n_eff;
      double rearth=6370949.0;

      R02=x0*x0+y0*y0;  //notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada
      h0=(sqrt((z0+rearth)*(z0+rearth) + R02 ) - rearth)/1.E3; //altitude of emission, in km

      rh0 = ns*exp(kr*h0); //refractivity at emission
      n_h0=1.E0+1.E-6*rh0; //n at emission

      hd=(groundz)/1.E3; //detector altitude

       // Vector from detector to average point on track. Making the integral in this way better guaranties the continuity
       // since the choping of the path will be always the same as you go farther away. If you start at your starting point, for a given geometry,
       // the choping points change with each starting position.

      ux = x0-xant;
      uy = y0-yant;        //the antenna position, considered to be at the core
      uz = z0-groundz;

      Rd=sqrt(ux*ux + uy*uy);
      kx=ux/Rd;
      ky=uy/Rd; //!k is a vector from the antenna to the track, that when multiplied by Rd will end in the track and sumed to antenna position will be equal to the track positon
      kz=uz/Rd;

//       integral starts at ground
      double nint=0;
      double sum=0.E0;
      double nextpx, nextpy, nextpz, nextR2, nexth;

      currpx=0.+xant;
      currpy=0.+yant;    //!current point (1st antenna position)
      currpz=groundz;
      currh=hd;

      while(Rd > stepsize){ //if distance projected on the xy plane is more than 10km
        nint=nint+1;
        nextpx=currpx+kx*stepsize;
        nextpy=currpy+ky*stepsize;           //this is the "next" point
        nextpz=currpz+kz*stepsize;

        nextR2=nextpx*nextpx + nextpy*nextpy; //!se usa el cuadrado, se puede ahorrar la raiz cuadrada
        nexth=(sqrt((nextpz+rearth)*(nextpz+rearth) + nextR2) - rearth)/1.E3;

        if(fabs(nexth-currh) > 1.E-10  ){   //check that we are not going at constant height, if so, the refraction index is constant
            sum=sum+(exp(kr*nexth)-exp(kr*currh))/(kr*(nexth-currh));
        }
        else{
            sum=sum+exp(kr*currh);
          }

        currpx=nextpx;
        currpy=nextpy;
        currpz=nextpz;  //Set new "current" point
        currh=nexth;

        Rd=Rd-stepsize; //reduce the remaining lenght
      }
      //enddo

      //when we arrive here, we know that we are left with the last part of the integral, the one closer to the track (and maybe the only one)

      nexth=h0;

      if(fabs(nexth-currh) > 1.E-10 ){ //check that we are not going at constant height, if so, the refraction index is constant
        sum=sum+(exp(kr*nexth)-exp(kr*currh))/(kr*(nexth-currh));
      }
      else{
        sum=sum+exp(kr*currh);
      }

      nint=nint+1;
      avn=ns*sum/nint;
      n_eff=1.E0+1.E-6*avn; //average (effective) n

  return n_eff;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double _ZHSEffectiveRefractionIndex(double x0,double y0,double z0, double xa,double ya, double za, double ns, double kr)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//ZHaires refractive index from Matias Tueros computation (more accurate function)
{
//subroutine effectiveindex(x0,y0,z0,xa,ya,za,n_eff,n_h0)
//
//    this routine computes the effective index between x0,y0,z0 and antenna a
//
//     InputParameters
//    [x,y,z]0     --> double precision: track position (m)
//     [x,y,z]a     --> double precision: antenna position (m)
//
//     Output parameters
//     n_eff        --> effective refraction index between x0,y0,z0 y antena na.
//     n_h0         --> index of refracion at x0,y0,z0 //useless here...
//
//

      //Declaration of arguments.
      double n_eff, n_h0;
      double rearth=6370949.0; // or 6371007.0 for the new aires

      //Declaration of internal variables
      int nint, iii ;
      double modr, ux, uy, uz, R02, h0, rh0, kx, ky, kz, avn, hd, sum ;
      double currpx, currpy, currpz, currh ;
      double nextpx, nextpy, nextpz, nextR2, nexth ;

      //Variable n integral calculation ///////////////////
      R02=x0*x0+y0*y0;
      h0=(sqrt( (z0+rearth)*(z0+rearth) + R02 ) - rearth)/1.e3 ;   //altitude of emission

      rh0=ns*exp(kr*h0) ; //refractivity at emission (this
      n_h0=1.e0+1.e-6*rh0 ; //n at emission //useless here...
      //std::cout<<"n_h0"<<n_h0<<ns<<kr<<x0<<y0<<injz-z0<<h0<<rh0<<std::endl;
      modr=sqrt(R02) ;

      if(modr > 1.e3){ // if inclined shower and point more than 20km from core. Using the core as reference distance is dangerous, its invalid in upgoing showers

        //Vector from average point of track to observer.
        ux = xa-x0;
        uy = ya-y0;
        uz = za-z0;

        //divided in nint pieces shorter than 10km
        nint=(modr/2.0e4)+1 ;
        kx=ux/nint ;
        ky=uy/nint ;       //k is vector from one point to the next
        kz=uz/nint ;

        currpx=x0 ;
        currpy=y0 ;        //current point (1st is emission point)
        currpz=z0 ;
        currh=h0 ;

        sum=0. ;
    for (iii=0; iii<nint; iii++){

      nextpx=currpx+kx ;
      nextpy=currpy+ky ; //this is the "next" point
      nextpz=currpz+kz ;
      nextR2=nextpx*nextpx + nextpy*nextpy ;
      nexth=(sqrt((nextpz+rearth)*(nextpz+rearth) + nextR2) - rearth)/1.e3 ;

      if(abs(nexth-currh) > 1.e-10  ){
        sum=sum+(exp(kr*nexth)-exp(kr*currh))/(kr*(nexth-currh)) ;
      }
      else{
        sum=sum+exp(kr*currh) ;
      }

      currpx=nextpx ;
      currpy=nextpy ;
      currpz=nextpz ; //Set new "current" point
      currh=nexth ;
      //std::cout<<"sum:"<<sum<<" i = "<<iii<<" nint = "<<nint<<std::endl;
    }

    avn=ns*sum/nint ;
    n_eff=1.e0+1.e-6*avn ; //average (effective) n
  }
  else{
    //withouth integral
    hd=za/1.e3 ; //detector altitude

    if(abs(hd-h0) < 1.e-10){
      avn=(ns/(kr*(hd-h0)))*(exp(kr*hd)-exp(kr*h0)) ;
      //std::cout<<"avn2:"<<avn<<std::endl;
    }
    else{
      avn=ns*exp(kr*h0) ;
      //std::cout<<"avn3:"<<avn<<std::endl;
      std::cout<<"Effective n: h0=hd"<<std::endl;
    }
    n_eff=1.e0+1.e-6*avn ; //average (effective) n
  }
///////////////////////////////////////////////////

  return n_eff;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double master_equation(double w_, double X_, double Delta_, double alpha_, double n0_, double n1_)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//Function of the Cerenkov angle equation
{
    double sa, saw;
    sa = sin(alpha_);
    saw = sin(alpha_ - w_);
    return X_*X_ * sa*sa * (n0_*n0_ - n1_*n1_) + 2.*Delta_ * X_ * sa * (n0_ - n1_*n1_*cos(w_))*saw + Delta_*Delta_ * (1. - n1_*n1_) * saw*saw;
  }

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void cross_product(double &vecx, double &vecy, double &vecz, double u1x, double u1y, double u1z, double u2x, double u2y, double u2z)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Procedure to compute the cross product of two vectors
{
vecx = u1y * u2z - u1z * u2y;
vecy = u1z * u2x - u1x * u2z;
vecz = u1x * u2y - u1y * u2x;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void compute_observer_position(double &x, double &y, double &z, double w, double kx, double ky, double kz, double ux, double uy, double uz, double x_Xmax, double y_Xmax, double z_Xmax, double GroundAltitude)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//Procedure to compute the observer position given a specific direction and angle and Xmax position
{
    double rotx, roty, rotz, Mxx, Mxy, Mxz, Myx, Myy, Myz, Mzx, Mzy, Mzz, dirx, diry, dirz, cw, sw;
    cw = cos(w);
    sw = sin(w);
    //compute the roation vector (perpendicular to k and u)
    std::cout <<"u "<<ux<<" "<<uy<<" "<<uz;
    std::cout <<"k "<<kx<<" "<<ky<<" "<<kz;
    cross_product(rotx, roty, rotz, ux, uy, uz, kx, ky, kz);
    //std::cout <<"rot "<<rotx<<" "<<roty<<" "<<rotz<<std::endl;
    rotx /= sqrt(rotx*rotx + roty*roty + rotz*rotz);
    roty /= sqrt(rotx*rotx + roty*roty + rotz*rotz);
    rotz /= sqrt(rotx*rotx + roty*roty + rotz*rotz);
    //compute the rotation matrix
    Mxx = cw + rotx*rotx*(1. - cw) ; Mxy = rotx*roty*(1. - cw) - rotz*sw ; Mxz = rotx*rotz*(1. - cw) + roty*sw ;
    Myx = roty*rotx*(1. - cw) + rotz*sw ; Myy = cw + roty*roty*(1. - cw) ; Myz = roty*rotz*(1. - cw) - rotx*sw ;
    Mzx = rotz*rotx*(1. - cw) - roty*sw ; Mzy = rotz*roty*(1. - cw) + rotx*sw ; Mzz = cw + rotz*rotz*(1. - cw) ;

    //Compute the new observer position
    dirx = Mxx*kx + Myx*ky + Mzx*kz;
    diry = Mxy*kx + Myy*ky + Mzy*kz;
    dirz = Mxz*kx + Myz*ky + Mzz*kz;

    double t = (GroundAltitude - z_Xmax) / dirz; // to check !!!
    x = x_Xmax+dirx*t;
    y = y_Xmax+diry*t;
    z = z_Xmax+dirz*t;
  }

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double Cerenkov_dichotomie(double w_start, double eta, double kx, double ky, double kz, double XmaxDist, double x_Xmax, double y_Xmax, double z_Xmax, double delta, double GroundAltitude)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Solve via dichotomie the Cerenkov angle
{

    double x, y, z, n0, n1;
    //compute the emission point before Xmax
    double x_Before = x_Xmax - kx * delta ;
    double y_Before = y_Xmax - ky * delta ;
    double z_Before = z_Xmax - kz * delta ;

    double k_planx = kx / sqrt(kx*kx + ky*ky); //x componant of the shower direction projected on the plane
    double k_plany = ky / sqrt(kx*kx + ky*ky); //x componant of the shower direction projected on the plane

    double ce = cos(eta);
    double se = sin(eta);

    //Compute the new observer position
    double ux = ce*k_planx + se*k_plany;
    double uy = - se*k_planx + ce*k_plany;

    double alpha = acos(kx*ux + ky*uy);

    double w_test = w_start;
    double w_step = 0.0001;
    double res = -1.e3;

    double it = 0;
    // clock_t begin = clock();
    while(res < 0){

        //Compute the test position
	//std::cout.precision(17);
        //std::cout<<"Inputs"<<" "<<w_test<<" "<<kx<<" "<<ky<<" "<<kz<<" "<<ux<<" "<<uy<<" "<<0<<" "<<x_Xmax<<" "<<y_Xmax<<" "<<z_Xmax<<" "<<GroundAltitude << std::endl;
        compute_observer_position( x,  y,  z,  w_test,  kx,  ky,  kz,  ux,  uy, 0, x_Xmax,  y_Xmax,  z_Xmax,  GroundAltitude);
	//std::cout<<"Observer position "<<x<<" "<<y<<" "<<z<<" "<<std::endl;
        n0 = _ZHSEffectiveRefractionIndex(x_Xmax, y_Xmax, z_Xmax, x, y, z, 325.0, -0.1218); //to checks here !!
        n1 = _ZHSEffectiveRefractionIndex(x_Before, y_Before, z_Before, x, y, z, 325.0, -0.1218); //to checks here !!
        res = master_equation(w_test, XmaxDist, delta, alpha, n0, n1);
	//std::cout<<"Master equation "<<w_test<<" "<<XmaxDist<<" "<<delta<<" "<<alpha<<" "<<n0<<" "<<n1<<" "<< res << std::endl;
        w_test += w_step;
        it++;
        if (it > 500){ //300 iteration -> angle > 2.8° largely above expected Cerenkov angle
            w_test = acos(1./GetRefractionIndexAtXmax(x_Xmax, y_Xmax, z_Xmax, 325.0, -0.1218)); //At least return the standard Cerenkov angle
            res = 1.;
          }
      }
      // clock_t end = clock();
      // unsigned long millis = (end -  begin) * 1000 / CLOCKS_PER_SEC;
      // std::cout<<"Time = "<<millis<<std::endl;

    return w_test - w_step;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void PSF_function_F( int& N, double* X, int& NF, double* F, int* Na,
  double* Xa, void* UFPARM )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  The objective function to minimise for the point source fit. It is built
//  as a chi2 over all antennas, requiring:
//
//  | Xa - Xs | - cs*( Ta - Ts ) = 0
//
//  Time is assumed to be expressed in m and speed cs is normalised to C0.
//  Note that although the evaluation of |Xa-Xs| is longer than the one of
//  of (Xa-Xs)^2 building the chi2 from |Xa-Xs| is more stable numericaly.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  static double d1, d2, d3, fi;

  std::cout.precision(15);
  F[ 0 ] = 0.0;
  for ( int i = 0; i < Na[ 0 ]; i++ )
  {
    d1      = ( X[ 0 ] - Xa[ i*N_ANTENNA_DATA     ] );
    d2      = ( X[ 1 ] - Xa[ i*N_ANTENNA_DATA + 1 ] );
    d3      = ( X[ 2 ] - Xa[ i*N_ANTENNA_DATA + 2 ] );
    fi      = X[ 4 ]*( X[ 3 ] - Xa[ i*N_ANTENNA_DATA + 3 ] ) + sqrt( d1*d1 + d2*d2 + d3*d3 );
    F[ 0 ] += fi*fi;
  }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void PSF_function_G( int& N, double* X, int& NF, double* G, int* Na,
  double* Xa, void* UFPARM )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  The gradient of the objective function.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  static double d1, d2, d3, d, fi;

  G[ 0 ] = G[ 1 ] = G[ 2 ] = G[ 3 ] = G[ 4 ] = 0.0;
  for ( int i = 0; i < Na[ 0 ]; i++ )
  {
    d1  = ( X[ 0 ] - Xa[ i*N_ANTENNA_DATA     ] );
    d2  = ( X[ 1 ] - Xa[ i*N_ANTENNA_DATA + 1 ] );
    d3  = ( X[ 2 ] - Xa[ i*N_ANTENNA_DATA + 2 ] );
    d   = sqrt( d1*d1 + d2*d2 + d3*d3 );
    fi  = X[ 4 ]*( X[ 3 ] - Xa[ i*N_ANTENNA_DATA + 3 ] ) + d;

    G[ 0 ] += ( X[ 0 ] - Xa[ i*N_ANTENNA_DATA	  ] )*fi/d;
    G[ 1 ] += ( X[ 1 ] - Xa[ i*N_ANTENNA_DATA + 1 ] )*fi/d;
    G[ 2 ] += ( X[ 2 ] - Xa[ i*N_ANTENNA_DATA + 2 ] )*fi/d;
    G[ 3 ] += X[ 4 ]*fi;
  }
  G[ 0 ] *= 2.0;
  G[ 1 ] *= 2.0;
  G[ 2 ] *= 2.0;
  G[ 3 ] *= 2.0;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void PWF_function_F( int& N, double* X, int& NF, double* F, int* Na,
  double* Xa, void* UFPARM )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  The objective function to minimise for plane wave fit. It is built
//  as a chi2 over all antennas, requiring:
//
//  ( Xa_j - Xa_i ).k - cr*( Ta_j - Ta_i ) = 0
//
//  over all pairs.
//
//  Time is assumed to be expressed in m and speed cr is normalised to C0
//
//  OMH 28/09/09: warning: due to modified conventions ( x=WE, y=SN ),
//  convertions from cartesian to spherical coordinates has been modified.
//
// MAJ 12/12/17 : GRAND convention (x=SN, y=EW) i.e. x_GRAND = y_TREND and y_GRAND = -x_TREND
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  static double st, ct, sp, cp, fi;

  ct = cos( X[ 0 ] );
  st = sin( X[ 0 ] );
  cp = cos( X[ 1 ] );
  sp = sin( X[ 1 ] );

  //double norm; //test

  F[ 0 ] = 0.0;
  for ( int j = 0; j < Na[ 0 ]-1; j++ )
  {
    for ( int i = j+1; i < Na[ 0 ]; i++ )
    {
    fi  = ( Xa[ j*N_ANTENNA_DATA     ] - Xa[ i*N_ANTENNA_DATA     ] )*st*cp;
    fi += ( Xa[ j*N_ANTENNA_DATA + 1 ] - Xa[ i*N_ANTENNA_DATA + 1 ] )*st*sp;
    fi += ( Xa[ j*N_ANTENNA_DATA + 2 ] - Xa[ i*N_ANTENNA_DATA + 2 ] )*ct;
    fi -= X[ 2 ]*( Xa[ j*N_ANTENNA_DATA + 3 ] - Xa[ i*N_ANTENNA_DATA + 3 ] );
    F[ 0 ]+= fi*fi;
    }
  }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void PWF_function_G( int& N, double* X, int& NF, double* G, int* Na,
  double* Xa, void* UFPARM )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  The gradient of the objective function.
//  OMH 28/09/09: warning: due to modified conventions (x=WE, y=SN),
//  conversions from cartesian to spherical corrdinates has been modified.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  static double st, ct, sp, cp, xij, yij, zij, tij, fi;

  //double norm;

  ct = cos( X[ 0 ] );
  st = sin( X[ 0 ] );
  cp = cos( X[ 1 ] );
  sp = sin( X[ 1 ] );

  G[ 0 ] = G[ 1 ] = G[ 2 ] = 0.0;
  for ( int j = 0; j < Na[ 0 ]-1; j++ )
    for ( int i = j+1; i < Na[ 0 ]; i++ )
  {
    //norm = sqrt(pow(Xa[ j*N_ANTENNA_DATA     ] - Xa[ i*N_ANTENNA_DATA     ], 2) + pow(Xa[ j*N_ANTENNA_DATA + 1 ] - Xa[ i*N_ANTENNA_DATA + 1 ], 2) + pow(Xa[ j*N_ANTENNA_DATA + 2 ] - Xa[ i*N_ANTENNA_DATA + 2 ], 2));
    xij = Xa[ j*N_ANTENNA_DATA     ] - Xa[ i*N_ANTENNA_DATA     ];
    yij = Xa[ j*N_ANTENNA_DATA + 1 ] - Xa[ i*N_ANTENNA_DATA + 1 ];
    zij = Xa[ j*N_ANTENNA_DATA + 2 ] - Xa[ i*N_ANTENNA_DATA + 2 ];
    tij = Xa[ j*N_ANTENNA_DATA + 3 ] - Xa[ i*N_ANTENNA_DATA + 3 ];

    fi  = ( xij*cp + yij*sp )*st + zij*ct - X[2]*tij;


		G[ 0 ]+= ( ( xij*cp + yij*sp )*ct - zij*st )*fi;
    //G[ 1 ]+= ( ( - xij*sp + yij*cp )*st + zij*ct )*fi;
		G[ 1 ]+= ( - xij*sp + yij*cp )*st*fi;
    G[ 2 ]-= tij*fi;
  }
  G[ 0 ]*= 2;
  G[ 1 ]*= 2;
  G[ 2 ]*= 2;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SWF_function_F( int& N, double* X, int& NF, double* F, int* Na,
  double* Xa, void* UFPARM )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  static double fi, dx_, dy_, dz_, x_xmax, y_xmax, z_xmax, _n;
  double GroundAltitude = 1086.;
  //; X[ 0 ] = theta ; X[ 1 ] = phi ; X[ 2 ] = r ; X[ 3 ] = t emission ; X[ 4 ] = c
  // Build Xmax position in opposite direction to the shower propagation
  x_xmax =  - X[ 2 ] * sin(X[ 0 ]) * cos(X[ 1 ]);
  y_xmax =  - X[ 2 ] * sin(X[ 0 ]) * sin(X[ 1 ]);
  z_xmax =  - X[ 2 ] * cos(X[ 0 ]) + GroundAltitude;

  F[ 0 ] = 0.0;
    for ( int i = 0; i < Na[ 0 ]; i++)
  {
      dx_ = Xa[ i*N_ANTENNA_DATA ] - x_xmax;
      dy_ = Xa[ i*N_ANTENNA_DATA + 1] - y_xmax;
      dz_ = Xa[ i*N_ANTENNA_DATA + 2] - z_xmax;
      _n = _ZHSEffectiveRefractionIndex(x_xmax, y_xmax, z_xmax, Xa[ i*N_ANTENNA_DATA ], Xa[ i*N_ANTENNA_DATA + 1], Xa[ i*N_ANTENNA_DATA + 2], 325.0, -0.1218);
      fi =  X[ 4 ]*(Xa[ i*N_ANTENNA_DATA + 3] - X[ 3 ]) - _n*sqrt(dx_*dx_ + dy_*dy_ + dz_*dz_);
      F[ 0 ] += fi*fi;
    }
    // std::cout<<"F[ 0 ]  = "<<F[ 0 ]<<" X[ 0 ] = "<<X[ 0 ]*180./M_PI<<" X[ 1 ] = "<<X[ 1 ]*180./M_PI<<" X[ 2 ] = "<<X[ 2 ]<<" X[ 3 ] = "<<X[ 3 ]<<" X[ 4 ] = "<<X[ 4 ]<<std::endl;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SWF_function_G( int& N, double* X, int& NF, double* G, int* Na,
  double* Xa, void* UFPARM )
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
  //to redo here... not used for minimisation
  static double st, ct, sp, cp, fi, xij, yij, zij;
  // X[ 0 ] = theta ; X[ 1 ] = phi ; X[ 2 ] = X_xmax ; X[ 3 ] = Y_xmax ; X[ 4 ] = Z_xmax ; X[ 5 ] = c
  ct = cos( X[ 0 ] );
  st = sin( X[ 0 ] );
  cp = cos( X[ 1 ] );
  sp = sin( X[ 1 ] );

  //
  G[ 0 ] = G[ 1 ] = G[ 2 ] = G[ 3 ] = G[ 4 ] = G[ 5 ] = 0.0;
  for ( int j = 0; j < Na[ 0 ]-1; j++ )
    for ( int i = 0; i < Na[ 0 ]-1; i++ )
  {
    //todo latter

  }
  G[ 0 ] *= 2;
  G[ 1 ] *= 2;
  G[ 2 ] *= 2;
  G[ 3 ] *= 2;
  G[ 4 ] *= 2;
  G[ 5 ] *= 2;

  std::cout<<"G[ 0 ]"<<G[ 0 ]<<std::endl;
  std::cout<<"G[ 1 ]"<<G[ 1 ]<<std::endl;
  std::cout<<"G[ 2 ]"<<G[ 2 ]<<std::endl;
  std::cout<<"G[ 3 ]"<<G[ 3 ]<<std::endl;
  std::cout<<"G[ 4 ]"<<G[ 4 ]<<std::endl;
  std::cout<<"G[ 5 ]"<<G[ 5 ]<<std::endl;

}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void ADF_function_F( int& N, double* X, int& NF, double* F, int* Na,
  double* Xa, void* UFPARM)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//Cerenkov pattern fit based on amplitudes measurments
{
  static double st, ct, sp, cp, fi, _k_x, _k_y, _k_z, _kxB_x, _kxB_y, _kxB_z, _kxkxB_x, _kxkxB_y, _kxkxB_z, norm, XmaxDist, _xa, _ya, _za, _xa_sp, _ya_sp, _za_sp, _l_antenna, _eta, _omega, _alpha, _cerenkov, _width, _asym, _adf, B_x, B_y, B_z, _xi;
  // X[ 0 ] = theta ; X[ 1 ] = phi ; X[ 2 ] = cerenkov_width ; X[ 3 ] = amplitude_corr ; X[ 4 ] = c
  ct = cos( X[ 0 ] );
  st = sin( X[ 0 ] );
  cp = cos( X[ 1 ] );
  sp = sin( X[ 1 ] );

  //To implement later (TODO: read B field from inputs)
  double GroundAltitude = 1086.;//2900.;
  double declination = 0.;
  double inclination = M_PI/2. + 1.0609856522873529;//1.0023425894203433;
  B_x = cos(declination) * sin(inclination);
  B_y = sin(declination)* sin(inclination);
  B_z = cos(inclination);

  //0) Compute shower frame (TODO: rewrite computation here)
  _k_x = st * cp;
  _k_y = st * sp;
  _k_z = ct;
  //std::cout<<"_k_x = "<<_k_x<<" _k_y = "<<_k_y<<" _k_z = "<<_k_z<<std::endl;
  _kxB_x = _k_y * B_z - _k_z * B_y;
  _kxB_y = _k_z * B_x - _k_x * B_z;
  _kxB_z = _k_x * B_y - _k_y * B_x;
  norm = sqrt(_kxB_x*_kxB_x + _kxB_y*_kxB_y + _kxB_z*_kxB_z);
  _kxB_x /= norm;
  _kxB_y /= norm;
  _kxB_z /= norm;
  //std::cout<<"_kxB_x = "<<_kxB_x<<" _kxB_y = "<<_kxB_y<<" _kxB_z = "<<_kxB_z<<std::endl;
  _kxkxB_x = _k_y * _kxB_z - _k_z * _kxB_y;
  _kxkxB_y = _k_z * _kxB_x - _k_x * _kxB_z;
  _kxkxB_z = _k_x * _kxB_y - _k_y * _kxB_x;
  norm = sqrt(_kxkxB_x*_kxkxB_x + _kxkxB_y*_kxkxB_y + _kxkxB_z*_kxkxB_z);
  _kxkxB_x /= norm;
  _kxkxB_y /= norm;
  _kxkxB_z /= norm;
  //std::cout<<"_kxkxB_x = "<<_kxkxB_x<<" _kxkxB_y = "<<_kxkxB_y<<" _kxkxB_z = "<<_kxkxB_z<<std::endl;

  XmaxDist = (GroundAltitude - Xa[ N_ANTENNA_DATA + 7 ])/_k_z; //to check !!!!
  _asym = 0.01* (1. - (_k_x*B_x+_k_y*B_y+_k_z*B_z)*(_k_x*B_x+_k_y*B_y+_k_z*B_z));
  //std::cout<<"XmaxDist = "<<XmaxDist<<" _asym = "<<_asym<<std::endl;

  //Computation of the Cerenkov angles for each direction in the footprint
  int n_angles = 19;
  double CerenkovTable[n_angles];
  double diff_old, diff_new, eta_table;
  for (int j = 0; j < n_angles; j++)
  {
    eta_table = float(10.*j*M_PI/180.);
    //std::cout<<"eta_table = "<<eta_table*180./M_PI<<std::endl;
    CerenkovTable[j] = Cerenkov_dichotomie( 0.,  eta_table,  _k_x,  _k_y,  _k_z,  XmaxDist,  Xa[ N_ANTENNA_DATA + 5],  Xa[ N_ANTENNA_DATA + 6],  Xa[ N_ANTENNA_DATA + 7],  2.e3,  GroundAltitude);
    //std::cout<<" CerenkovTable[j] = "<<CerenkovTable[j]<<std::endl;
  }

  F[ 0 ] = 0.0;
    for ( int i = 0; i < Na[ 0 ]; i++)
  {

      //1) Antenna position from Xmax
      _xa = (Xa[ i*N_ANTENNA_DATA ] - Xa[ i*N_ANTENNA_DATA + 5]);
      _ya = (Xa[ i*N_ANTENNA_DATA + 1] - Xa[ i*N_ANTENNA_DATA + 6]);
      _za = (Xa[ i*N_ANTENNA_DATA + 2] - Xa[ i*N_ANTENNA_DATA + 7]);

      //2) Antenna position in shower frame (at Xmax position !)
      _xa_sp = _xa*_kxB_x + _ya*_kxB_y + _za*_kxB_z;
      _ya_sp =  _xa*_kxkxB_x + _ya*_kxkxB_y + _za*_kxkxB_z;
      _za_sp =  _xa*_k_x + _ya*_k_y + _za*_k_z;
      //std::cout<<"_xa = "<<_xa<<" _ya = "<<_ya<<" _za = "<<_za<<std::endl;
      //std::cout<<"_xa_sp = "<<_xa_sp<<" _ya_sp = "<<_ya_sp<<" _za_sp = "<<_za_sp<<std::endl;

      //2) Compute new coordinates
      _l_antenna = sqrt(_xa*_xa + _ya*_ya + _za*_za);
      _eta = atan2(_ya_sp,_xa_sp);
      _omega = acos((_k_x*_xa + _k_y*_ya + _k_z*_za) / _l_antenna);
      _alpha = acos(_za/_l_antenna);
      _xi = M_PI - acos((-_k_x*(_xa/_l_antenna - _k_x) - _k_y*(_ya/_l_antenna - _k_y))/(sqrt((_xa/_l_antenna - _k_x)*(_xa/_l_antenna - _k_x) + (_ya/_l_antenna - _k_y)*(_ya/_l_antenna - _k_y))*sqrt(_k_x*_k_x + _k_y*_k_y)));
      //std::cout<<"_l_antenna = "<<_l_antenna<<" _eta = "<<_eta*180./M_PI<<" _omega = "<<_omega*180./M_PI<<" _alpha = "<<_alpha*180./M_PI<<" _xi = "<<_xi*180./M_PI<<std::endl;

      //3)Find the Cerenkov value corresponding to the eta angle of the antenna
      diff_old = fabs(_xi+M_PI);
      _cerenkov = 0;
      for (int j = 0; j < n_angles; j++){
        diff_new = fabs(_xi - 10*M_PI/180.*j);
        if(diff_new<diff_old){
          _cerenkov = CerenkovTable[j];
        }
        diff_old = diff_new;
      }
      //std::cout<<"_cerenkov = "<<_cerenkov<<std::endl;

      _width = ct/cos(_alpha) * X[ 2 ];
      //std::cout<<" _width = "<<_width<<std::endl;

      //4) Angular Distribution Function (ADF)
      _adf = X [ 3 ]/_l_antenna / (1. + 4. * ((((tan(_omega)/tan(_cerenkov))*(tan(_omega)/tan(_cerenkov)) - 1.)/_width)*(((tan(_omega)/tan(_cerenkov))*(tan(_omega)/tan(_cerenkov)) - 1.)/_width))) * (1. + _asym*cos(_eta)) ;
      if (isnan(_adf)){std::cout<<"_adf = "<<_adf<<std::endl;}
      //std::cout<<"omega = "<<_omega<<" f"<<1./ (1. + 4. * ((((tan(_omega)/tan(_cerenkov))*(tan(_omega)/tan(_cerenkov)) - 1.)/_width)*(((tan(_omega)/tan(_cerenkov))*(tan(_omega)/tan(_cerenkov)) - 1.)/_width)))<<std::endl;

      //5) Chi square
      fi = Xa[ i*N_ANTENNA_DATA + 4] - _adf;
      //std::cout<<"_adf = "<<_adf<<" Xa[ i*N_ANTENNA_DATA + 4] = "<<Xa[ i*N_ANTENNA_DATA + 4]<<" fi = "<<fi<<std::endl;
      F[ 0 ] += fi*fi;
    }
    if (isnan(F[ 0 ])){std::cout<<"F[ 0 ] = "<<F[ 0 ]<<" X[ 0 ] = "<<X[ 0 ]*180./M_PI<<" X[ 1 ] = "<<X[ 1 ]*180./M_PI<<" X[ 2 ] = "<<X[ 2 ]<<" X[ 3 ] = "<<X[ 3 ]<<" X[ 4 ] = "<<X[ 4 ]<<std::endl;}
    //std::cout<<"F[ 0 ] = "<<F[ 0 ]<<" X[ 0 ] = "<<X[ 0 ]*180./M_PI<<" X[ 1 ] = "<<X[ 1 ]*180./M_PI<<" X[ 2 ] = "<<X[ 2 ]<<" X[ 3 ] = "<<X[ 3 ]<<" X[ 4 ] = "<<X[ 4 ]<<std::endl;

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void ADF_function_G( int& N, double* X, int& NF, double* G, int* Na,
  double* Xa, void* UFPARM)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//Cerenkov pattern fit based on amplitudes measurments
{
  static double st, ct, sp, cp, fi, dx_, dy_, dz_, angle_, lorentz_func;
  // X[ 0 ] = theta ; X[ 1 ] = phi ; X[ 2 ] = cerenkov_angle ; X[ 3 ] = cerenkov_width ; X[ 6 ] = c
  ct = cos( X[ 0 ] );
  st = sin( X[ 0 ] );
  cp = cos( X[ 1 ] );
  sp = sin( X[ 1 ] );

    G[ 0 ] = G[ 1 ] = G[ 2 ] = G[ 3 ] = G[ 4 ] = G[ 5 ] = 0.0;
  //   for ( int i = 0; i < Na[ 0 ]; i++)
  // {
  //     dx_ = Xa[ i*N_ANTENNA_DATA ] - Xa[ i*N_ANTENNA_DATA + 5];
  //     dy_ = Xa[ i*N_ANTENNA_DATA + 1] - Xa[ i*N_ANTENNA_DATA + 6];
  //     dz_ = Xa[ i*N_ANTENNA_DATA + 2] - Xa[ i*N_ANTENNA_DATA + 7];
  //
  //     fi = Xa[ i*N_ANTENNA_DATA + 4] ;
  //   }
  //   //Finish the derivative fit function -> numerical derivative until done

    G[ 0 ] *= 2;
    G[ 1 ] *= 2;
    G[ 2 ] *= 2;
    G[ 3 ] *= 2;
    G[ 4 ] *= 2;
    G[ 5 ] *= 2;
}
