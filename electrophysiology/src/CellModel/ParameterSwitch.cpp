/*
 * File: ParameterSwitch.cpp
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


#include <ParameterSwitch.h>

ParameterSwitch::ParameterSwitch(vbNewElphyParameters *s, unsigned int vtLASTEntry) {
  // cerr<<"ParaSwitch() mit vbNewElphyParameters, last="<<vtLASTEntry<<"\n";
  stat             = s;
  vtLAST           = vtLASTEntry;
  useDynamicValues = false;
}

ML_CalcType ParameterSwitch::getValue(int vt) {
  // ElphyParameter v=stat->P[vt];
  /*
      int dynVar=stat->P[vt].dynamicVar;
      ML_CalcType value=stat->P[vt].value;
      return useDynamicValues?(dynVar<0?value:dyn[dynVar]):value;
   */
  return useDynamicValues ? (stat->P[vt].dynamicVar <
                             0 ? stat->P[vt].value : dyn[stat->P[vt].dynamicVar]) : stat->P[vt].value;

  // cerr<<"getValue: stat->K_o"<<stat->K_o<<", *stat->K_o="<<*(stat->K_o)<<endl;
  // cerr<<"dynamicVar="<<stat->P[vt].dynamicVar<<endl;
  if (stat->P[vt].dynamicVar < 0) {
    // cerr<<"static ...\n";
    return stat->P[vt].value;
  } else {
    cerr<<"dynamic ...\t(staticValue="<<stat->P[vt].value<<")\n";
    if ((int)dyn.size() < stat->P[vt].dynamicVar) {
      cerr<<"undefined in this parameterset - using static value ...\n";
      return stat->P[vt].value;
    } else {
      return dyn[stat->P[vt].dynamicVar];
    }
  }
}

bool ParameterSwitch::addDynamicParameter(Parameter pDynPara) {
  // cerr<<"bisher sind "<<dyn.size()<<" dynamische Parameter angelegt ...\n";
  dyn.push_back(pDynPara.value);
  unsigned int index = vtFirst;
  for (unsigned int y = vtFirst; y < vtLAST; y++) {
    if (stat->P[y].name == pDynPara.name) {
      stat->P[y].dynamicVar = dyn.size()-1;

      // cerr<<y<<" -> "<<dyn.size()-1<<endl;
      index = y;
    }
  }

  if (index == vtFirst) {
    throw kaBaseException("Parameter '%s' was not defined in the current implementation of the model!",
                          pDynPara.name.c_str());
  }

  // cerr<<"jetzt sind "<<dyn.size()<<" dynamische Parameter angelegt ...\n";
  // for (int x=0;x<dyn.size();x++){
  // cerr<<"\t"<<x<<": "<<(float)dyn[x]<<endl;
  // }
  // if (!useDynamicValues)
  //    cerr<<"using dynamic values is set!\n";
  useDynamicValues = true;
  /*if (dyn[stat->P[index].dynamicVar]!=pDynPara.value)
      cerr<<dyn[stat->P[index].dynamicVar]<<"!="<<pDynPara.value<<endl;*/
  return dyn[stat->P[index].dynamicVar] == pDynPara.value;
}  // ParameterSwitch::addDynamicParameter
