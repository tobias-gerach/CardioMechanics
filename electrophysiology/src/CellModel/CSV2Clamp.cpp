/*
 * File: CSV2Clamp.cpp
 *
 * Institute of Biomedical Engineering, 
 * Karlsruhe Institute of Technology (KIT)
 * https://www.ibt.kit.edu
 * 
 * Repository: https://github.com/KIT-IBT/CardioMechanics
 *
 * License: GPL-3.0 (See accompanying file LICENSE or visit https://www.gnu.org/licenses/gpl-3.0.html)
 *
 */


#include <kaVersion.h>

using namespace std;

int main(int argc, char *const argv[]) {
  kaVersion vers(1, 0, 0, argc, argv);

  vers.addOption("", "inputfile", OT_required, "", "either JSim, OpenCor, or OpenCell file");
  vers.addOption("", "outputfile", OT_optional, "inputfile.clamp", "outputfile in cell model clamp file format");
  vers.addOption("c", "vcolumn", OT_optional, "1", "Zero-based index of column containing the voltage");
  char *pcenvfile;
  char *clampfile;
  char  zeile[4096];
  char  t1[30], t2[30], v[30];
  int   line = 0, j = 0, k = 0, l = 0, m = 0, l_read = 0, l_write = 0, t_write = 0;
  int   pcenv;
  char  ch = 'a', c_open;
  double time1 = 0.0, dt, vm;

  int column = 1;

  for (int i = 0; i < 30; i++) {
    v[i] = 0;
  }
  for (int i = 0; i < 30; i++) {
    t1[i] = 0;
  }
  for (int i = 0; i < 30; i++) {
    t2[i] = 0;
  }

  if (argc < 2) {  // Not enough parameters
    vers.printHelp();
    return 1;
  }

  char pread = 0;
  for (int i = 1; i < argc; i++) {
    if ((argv[i][0] == '-') && (argv[i][1] == 'c') ) {
      i++;
      if (i == argc) {
        vers.printHelp();
        cerr << "#Missing value for parameter -c, exit." << endl;
        exit(-1);
      }
      column = atoi(argv[i]);
    } else if (pread == 0) {      // No file set until now
      pcenvfile = new char[strlen(argv[i])];
      strcpy(pcenvfile, argv[i]);
      pread++;
    } else if (pread == 1) {      // Inputfile already set
      clampfile = new char[strlen(argv[i])];
      strcpy(clampfile, argv[i]);
      pread++;
    } else {
      cerr << "#Parameter ignored: " << argv[i] << endl;
    }
  }

  switch (pread) {
    case 0:  // No inputfile has been set
      vers.printHelp();
      cerr << "#No inputfile specified. Exit." << endl;
      exit(-1);
      break;
    case 1:  // Only inputfile has been set -> use inputfile.clamp as output file
      clampfile = new char[strlen(pcenvfile)+1];
      strcpy(clampfile, pcenvfile);
      clampfile[strlen(pcenvfile)-3] = 'c';
      clampfile[strlen(pcenvfile)-2] = 'l';
      clampfile[strlen(pcenvfile)-1] = 'a';
      clampfile[strlen(pcenvfile)]   = 'm';
      clampfile[strlen(pcenvfile)+1] = 'p';
      break;
    default:  // Outputfile has been set
      break;
  }

  /*if (argc == 2) {
        strcpy(pcenvfile,argv[1]);
        strcpy(clampfile,argv[1]);
        clampfile[strlen(pcenvfile)-3]='c';
        clampfile[strlen(pcenvfile)-2]='l';
        clampfile[strlen(pcenvfile)-1]='a';
        clampfile[strlen(pcenvfile)]='m';
        clampfile[strlen(pcenvfile)+1]='p';
     }
     else {
        strcpy(pcenvfile,argv[1]);
        strcpy(clampfile,argv[2]);
     }  */

  ofstream ausgabe;
  ausgabe.open(clampfile, ios_base::in);
  ausgabe.close();
  if (ausgabe.good()) {
    cout << "#overwrite " <<  clampfile << " (y/n)? : ";
    cin >> c_open;
    if (c_open != 'y')
      return 1;
  } else {}

  ifstream eingabe(pcenvfile, ios::in);
  if (eingabe.good()) {
    eingabe.seekg(0L, ios::beg);
    for (int i = 0; i < 5; i++) {
      eingabe.get(ch);
    }
    if (ch == '(') {
      pcenv = 1;
      cout << "#found OpenCell file" << endl;
    } else if (ch == 'r') {
      pcenv = 2;
      cout << "#found OpenCor file" << endl;
    } else if (ch == ',') {
      pcenv = 0;
      cout << "#found jsim file" << endl;
    } else {
      cout << "#unknown file" << endl;
      eingabe.close();
      return 1;
    }

    // calculate dt for jsim file
    eingabe.seekg(0L, ios::beg);
    if (pcenv == 0) {
      while (!eingabe.eof()) {
        eingabe.getline(zeile, 4096);
        if (line == 1) {
          j = 0;
          while (zeile[j] != ',') {
            t1[j] = zeile[j];
            j++;
          }
        }
        if (line == 2) {
          j = 0;
          while (zeile[j] != ',') {
            t2[j] = zeile[j];
            j++;
          }
          j = 0;
          break;
        }
        line++;
      }
      dt = 0.001* (atof(t2) - atof(t1));

      // save Vm for jsim file
      ofstream ausgabe(clampfile, ios_base::out);
      cout << "#saving to " << clampfile << "..." << flush;
      eingabe.seekg(0L, ios::beg);
      line = 1;
      while (!eingabe.eof()) {
        eingabe.getline(zeile, 4096);
        if (line > 1) {
          for (int l = 0; l < strlen(zeile); l++) {
            if (zeile[l] == ',')
              k++;
            if ((k == column) && (zeile[l] != ',') ) {
              v[m] = zeile[l];
              m++;
            }
            if (k > column)
              break;
          }
          m  = 0;
          k  = 0;
          vm = 0.001*atof(v);
          if (!eingabe.eof()) {
            ausgabe.precision(8);
            ausgabe << "Vm " << dt << " ";
            ausgabe.precision(16);
            ausgabe << vm << "\n";
          }
          for (int i = 0; i < 20; i++) {
            v[i] = 0;
          }
        }
        line++;
      }
      cout << "done" << endl;
      eingabe.close();
      ausgabe.close();
    } else {
      // save Vm for OpenCell file
      cout << "#saving to " << clampfile << "..." << flush;
      ofstream ausgabe(clampfile, ios_base::out);
      eingabe.seekg(0L, ios::beg);
      line   = 0;
      l_read = 0;
      j      = 0;
      while (eingabe.get(ch)) {
        if (ch == '\n') {
          line++;
          l = 0;
          k = 0;
          j = 0;
        }
        if (ch == ',')
          k++;
        if ((ch != ',') && (k == 0) && (line > 1) ) {
          t1[l] = ch;
          l++;
        }
        if ((ch != ',') && (k == column) ) {
          v[j] = ch;
          j++;
          l_write = 1;
        }
        if ((k > column) && (l_write == 1) && (line > 0) ) {
          vm      = 0.001*atof(v);
          t_write = 1;
          l_write = 0;
        }
        if ((k == 1) && (t_write == 1) ) {
          double time2 = atof(t1);
          dt    = (time2-time1) * (pcenv == 1 ? 1.0 : 0.001);
          time1 = time2;
          if (!eingabe.eof()) {
            ausgabe.precision(8);
            ausgabe << "Vm " << dt << " ";
            ausgabe.precision(16);
            ausgabe << vm << "\n";
            t_write = 0;
          }
          for (int i = 0; i < 30; i++) {
            v[i] = 0;
          }
          for (int i = 0; i < 30; i++) {
            t1[i] = 0;
          }
        }
      }
      cout << "done" << endl;
      eingabe.close();
      ausgabe.close();
    }
  } else {cout << "#file not found" << endl;}
  return 0;
}  // main

/*! \page CSV2Clamp CSV2Clamp
   Translates an action potential generated by JSim or OpenCell into a clamp protocol

   \section SYNOPSIS_CSV2Clamp SYNOPSIS
   CSV2Clamp \<inputfile\> \n
   [\<outputfile\>]\n
   [-version]\n

   \section OPTIONS_CSV2Clamp OPTIONS
   \param "<inputfile>" Either JSim, OpenCor, or OpenCell file containing transmembrane voltage information
   \param "<outputfile>" (default: inputfile.clamp): outputfile in cell model clamp file format
   \param "-version" Print version information

   \section DESCRIPTION_CSV2Clamp DESCRIPTION
   Translates an action potential generated by JSim, OpenCor or OpenCell into a clamp protocol

   \section SOURCE_CSV2Clamp SOURCE
   CSV2Clamp.cpp

   \section SEEALSO_CSV2Clamp SEE ALSO
   \ref ElphyModelTest

   \section CHANGELOG_CSV2Clamp CHANGELOG
   V1.0.0 - 22.09.2008 (Gunnar Seemann): Initial programming\n
   V1.0.1 - 22.09.2008 (Gunnar Seemann): OpenCor added. OpenCell/OpenCor can now handle unequal time steps\n
 */
