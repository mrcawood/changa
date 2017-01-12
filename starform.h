/// @file starform.h
/// Declarations for star formation.

#ifndef STARFORM_HINCLUDED
#define STARFORM_HINCLUDED

#include "parameters.h"
#include "imf.h"

/// Parameters and methods to implement star formation.
class Stfm : public PUP::able  {
 private:
    Stfm& operator=(const Stfm& st);
    double dGmUnit;		/* system mass in grams */
    double dMsolUnit;    /* system mass unit in Msol */
    double dGmPerCcUnit;	/* system density in gm/cc */
    double dSecUnit;		/* system time in seconds */
    double dErgUnit;		/* system energy in ergs */
    double dPhysDenMin;		/* Physical density minimum for star
				   formation (in system units) */
    double dOverDenMin;		/* Overdensity minimum for star formation */
    double dTempMax;		/* Form stars below this temperature
				   EVEN IF the gas is not cooling. */
    double dSoftMin;		/* Jean's length as a fraction of
				   softening at which to form stars*/
    double dCStar;		/* Star formation constant */
    double dStarEff;		/* Fraction of gas mass converted into
				 star mass per timestep. */
    double dInitStarMass;       /* Fixed Initial Star Mass */
    double dMinSpawnStarMass;   /* Minimum Initial Star Mass */
    double dMaxStarMass;	/* maximum mass star particle to form */
    int bGasCooling;		/* Can we call cooling for temperature */
 public:
    int bUseStoch;          /* use stochastic IMF */
    int iStarFormRung;		/* rung for star formation */
    int iRandomSeed;		/* seed for probability */
    double dMinGasMass;		/* minimum mass gas before we delete
				   the particle. */
    double dDeltaStarForm;	/* timestep in system units */
    void AddParams(PRM prm);
    void CheckParams(PRM prm, struct parameters &param);
    bool isStarFormRung(int aRung) {return aRung <= iStarFormRung;}
    GravityParticle *FormStar(GravityParticle *p,  COOL *Cool, double dTime,
			      double dDelta, double dCosmoFac, double *T);
    IMF *imf = new Kroupa01();

    Stfm() {}
    PUPable_decl(Stfm);
    Stfm(const Stfm& st);
    Stfm(CkMigrateMessage *m) : PUP::able(m) {}
    ~Stfm() {
        delete imf;
    }
    inline void pup(PUP::er &p);

    };

// "Deep copy" constructer is needed because of imf pointer
inline Stfm::Stfm(const Stfm& st) {
    dMsolUnit = st.dMsolUnit;
    dGmUnit = st.dGmUnit;
    dGmPerCcUnit = st.dGmPerCcUnit;
    dSecUnit = st.dSecUnit;
    dErgUnit = st.dErgUnit;
    dPhysDenMin = st.dPhysDenMin;
    dOverDenMin = st.dOverDenMin;
    dTempMax = st.dTempMax;
    dSoftMin = st.dSoftMin;
    dCStar = st.dCStar;
    dStarEff = st.dStarEff;
    dInitStarMass = st.dInitStarMass;
    dMinSpawnStarMass = st.dMinSpawnStarMass;
    dMaxStarMass = st.dMaxStarMass;
    bGasCooling = st.bGasCooling;
    bUseStoch = st.bUseStoch;
    iStarFormRung = st.iStarFormRung;
    iRandomSeed = st.iRandomSeed;
    dMinGasMass = st.dMinGasMass;
    dDeltaStarForm = st.dDeltaStarForm;
    imf = st.imf->clone();
}

inline void Stfm::pup(PUP::er &p) {
    p|bUseStoch;
    p|dDeltaStarForm;
    p|iStarFormRung;
    p|iRandomSeed;
    p|dMsolUnit;
    p|dGmUnit;
    p|dGmPerCcUnit;
    p|dSecUnit;
    p|dErgUnit;
    p|dPhysDenMin;
    p|dOverDenMin;
    p|dTempMax;
    p|dSoftMin;
    p|dCStar;
    p|dStarEff;
    p|dInitStarMass;
    p|dMinSpawnStarMass;
    p|dMinGasMass;
    p|dMaxStarMass;
    p|bGasCooling;
    p|imf;
    }

/** @brief Holds statistics of the star formation event */
class StarLogEvent
{
 public:
    int64_t iOrdStar;
    int64_t iOrdGas;
    double timeForm;
    Vector3D<double> rForm;
    Vector3D<double> vForm;
    double massForm;
    double rhoForm;
    double TForm;
 StarLogEvent() : iOrdGas(-1),	timeForm(0),rForm(0),vForm(0),
	massForm(0),rhoForm(0),TForm(0){}
    StarLogEvent(GravityParticle *p, double dCosmoFac, double TempForm) {
	iOrdGas = p->iOrder;
	// star's iOrder assigned in TreePiece::NewOrder
	timeForm = p->fTimeForm();
	rForm = p->position;
	vForm = p->velocity;
	massForm = p->fMassForm();
	rhoForm = p->fDensity/dCosmoFac;
	TForm = TempForm;
	}
    void pup(PUP::er& p) {
	p | iOrdStar;
	p | iOrdGas;
	p | timeForm;
        p | rForm;
	p | vForm;
	p | massForm;
	p | rhoForm;
	p | TForm;
	}
    };

/** @brief Log of star formation events to be written out to a file */
class StarLog : public PUP::able
{
 public:
    int nOrdered;		/* The number of events that have been
				   globally ordered, incremented by
				   pkdNewOrder() */
    std::string fileName;
    std::vector<StarLogEvent> seTab;		/* The actual table */
 StarLog() : nOrdered(0),fileName("starlog") {}
    void flush();
    PUPable_decl(StarLog);
 StarLog(CkMigrateMessage *m) : PUP::able(m) {}
    void pup(PUP::er& p) {
	PUP::able::pup(p);
	p | nOrdered;
	p | fileName;
	p | seTab;
	}
    };

#endif
