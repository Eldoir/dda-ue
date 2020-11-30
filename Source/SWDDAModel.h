#pragma once

#include <CoreMinimal.h>

#include "SWDDAModel.generated.h"

class USWDDADataManager;
class USWDDAAttempt;
class USWModelLR;

UENUM(BlueprintType)
enum class ESWDDAAlgorithm : uint8
{
    DDA_LOGREG, //Utilise la regression logistique (si modèle calibré, sinon PM_DELTA)
    DDA_PMDELTA, //Si on gagne, theta monte, si on perds, theta descend
    DDA_RANDOM_THETA, //Choisit un theta random
    DDA_RANDOM_LOGREG //Choisit une diff random et en déduit le theta avec la logreg (sinon on fait random theta)
};

UENUM(BlueprintType)
enum class ESWDDALogRegError : uint8
{
    OK,
    NOT_ENOUGH_SAMPLES,
    NOT_ENOUGH_WINS,
    NOT_ENOUGH_FAILS,
    NEWTON_RAPHSON_ERROR,
    ACCURACY_TOO_LOW,
    SUM_ERROR_TOO_HIGH,
    SD_PRED_TOO_LOW,
    SUM_ERROR_IS_NAN,
    SD_PRED_IS_NAN
};

USTRUCT(BlueprintType)
struct FSWDiffParams
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float TargetDiff;
    UPROPERTY(BlueprintReadOnly)
    float TargetDiffWithExplo;
    UPROPERTY(BlueprintReadOnly)
    bool LogRegReady;
    UPROPERTY(BlueprintReadOnly)
    ESWDDALogRegError LogRegError;
    UPROPERTY(BlueprintReadOnly)
    float LRAccuracy;
    UPROPERTY(BlueprintReadOnly)
    float Theta;
    UPROPERTY(BlueprintReadOnly)
    int NbAttemptsUsedToCompute;
    UPROPERTY(BlueprintReadOnly)
    ESWDDAAlgorithm AlgorithmActuallyUsed;
    UPROPERTY(BlueprintReadOnly)
    ESWDDAAlgorithm AlgorithmWanted;
    UPROPERTY(BlueprintReadOnly)
    TArray<float> Betas;
};

UCLASS(Blueprintable)
class SWARMS_API USWDDAModel : public UObject
{
    GENERATED_BODY()
    
public:  
    /**
    * One can only create a model for a specific challenge and player, and with a chosen data mgmt strategy
    */
    UFUNCTION(BlueprintCallable)
    void Init( USWDDADataManager * dataManager, FString playerId, FString challengeId );

    /**
    * Allows to specify which difficulty adaptation algorithm we want the model to use (see enum description)
    */
    UFUNCTION(BlueprintCallable)
    void setDdaAlgorithm( ESWDDAAlgorithm algorithm );

    /**
    * Permet de déterminer un point de départ. L'algo pmdelta
    * va partir de la pour augmenter ou diminuer la difficulté
    */
    UFUNCTION(BlueprintCallable)
    void setPMInit(float lastTheta, bool wonLastTime = false);

    /**
    * Add new attempt to data and set is as last attempt
    */
    UFUNCTION(BlueprintCallable)
    void addLastAttempt(USWDDAAttempt * attempt);

    /**
    * Get gameplay parameter value for desired target difficulty
    * uses PMDeltaLastTheta for PMDelta algorithm
    */
    UFUNCTION(BlueprintCallable)
    FSWDiffParams computeNewDiffParams(float targetDifficulty, bool doNotUpdateLRAccuracy = false);

    UFUNCTION(BlueprintPure)
    bool checkDataAgainst( UPARAM(ref) TArray<USWDDAAttempt *> & attempts) const;

    //Settings Data
    FString PlayerId;
    FString ChallengeId;

    //Log reg model
    float LRAccuracy = 0;
    
    //PMDelta model
    bool PMInitialized = false;
    float PMLastTheta = 0;
    bool PMWonLastTime = false;
    float PMDeltaValue = 0.1;
    float PMDeltaExploMin = 0.5f;
    float PMDeltaExploMax = 1.0f;
    
    ESWDDAAlgorithm Algorithm;

private:
    //Settings Data
    UPROPERTY()
    USWDDADataManager * DataManager;

    //Log reg model
    UPROPERTY()
    USWModelLR * LogReg;
    const float LRMinimalAccuracy = 0.6;
    float LRExplo = 0.05f;
    bool LRAccuracyUpToDate = false;
    const int LRNbLastAttemptsToConsider = 150;
};