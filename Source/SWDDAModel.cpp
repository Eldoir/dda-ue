#include "SWDDAModel.h"

#include "SWDataLR.h"
#include "SWDDAAttempt.h"
#include "SWDDADataManager.h"
#include "SWLogisticRegression.h"
#include "SWModelLR.h"

void USWDDAModel::Init( USWDDADataManager * dataManager, const FString playerId, const FString challengeId )
{
    DataManager = dataManager;
    PlayerId = playerId;
    ChallengeId = challengeId;

    Algorithm = ESWDDAAlgorithm::DDA_LOGREG;
}

void USWDDAModel::setDdaAlgorithm( const ESWDDAAlgorithm algorithm )
{
    Algorithm = algorithm;
}

void USWDDAModel::setPMInit( const float lastTheta, const bool wonLastTime )
{
    PMLastTheta = lastTheta;
    PMWonLastTime = wonLastTime;
    PMInitialized = true;
}

void USWDDAModel::addLastAttempt( USWDDAAttempt * attempt )
{
    DataManager->addAttempt( PlayerId, ChallengeId, attempt );
    LRAccuracyUpToDate = false;
    PMWonLastTime = attempt->Result > 0;
    PMLastTheta = attempt->Thetas[ 0 ];
}

FSWDiffParams USWDDAModel::computeNewDiffParams( float targetDifficulty, const bool doNotUpdateLRAccuracy )
{
    FSWDiffParams diffParams;
    diffParams.LogRegReady = true;
    diffParams.AlgorithmWanted = Algorithm;
    diffParams.LogRegError = ESWDDALogRegError::OK;

    //Loading data
    auto attempts = DataManager->getAttempts( PlayerId, ChallengeId, LRNbLastAttemptsToConsider );

    //Data translation for LR
    auto * data = NewObject<USWDataLR>();
    TArray<TArray<float> > indepVars;
    TArray<float> depVars;
    for ( auto * attempt : attempts )
    {
        indepVars.Add( attempt->Thetas );
        depVars.Add( attempt->Result );
    }
    data->LoadDataFromList( indepVars, depVars );

    //On met a jour le dernier theta en fonction des datas si on ne l'a pas deja set
    if ( indepVars.Num() > 0 && !PMInitialized )
    {
        PMLastTheta = indepVars[ indepVars.Num() - 1 ][ 0 ];
        PMWonLastTime = depVars[ depVars.Num() - 1 ] > 0 ? true : false;
        PMInitialized = true;
    }

    //Check if enough data to update LogReg
    if ( attempts.Num() < 10 )
    {
        //Debug.Log( "Less than 10 attempts, can not use LogReg prediciton" );
        diffParams.LogRegReady = false;
        diffParams.LogRegError = ESWDDALogRegError::NOT_ENOUGH_SAMPLES;
    }
    else
    {
        //Chekcing wins and fails
        float nbFail = 0;
        float nbWin = 0;
        for ( auto * attempt : attempts )
        {
            if ( attempt->Result == 0 )
                nbFail++;
            else
                nbWin++;
        }

        //If only three fails or three wins
        if ( nbFail <= 3 || nbWin <= 3 )
        {
            //Debug.Log( "Less than 4 wins or 4 fails, will not use LogReg" );
            diffParams.LogRegReady = false;

            if ( nbWin <= 3 )
                diffParams.LogRegError = ESWDDALogRegError::NOT_ENOUGH_WINS;
            if ( nbFail <= 3 )
                diffParams.LogRegError = ESWDDALogRegError::NOT_ENOUGH_FAILS;
        }
    }

    if ( diffParams.LogRegReady )
    {
        //Debug.Log("Using " + data.DepVar.Length + " lines to update model");

        if ( !doNotUpdateLRAccuracy && !LRAccuracyUpToDate )
        {
            //Ten fold cross val
            LRAccuracy = 0;

            for ( int i = 0; i < 10; i++ )
            {
                float AccuracyNow = 0;
                data = data->shuffle();
                int nk = 10;
                for ( int k = 0; k < nk; k++ )
                {
                    auto * dataTrain = NewObject<USWDataLR>();
                    auto * dataTest = NewObject<USWDataLR>();
                    data->split( k * ( 100 / nk ), ( k + 1 ) * ( 100 / nk ), dataTrain, dataTest );
                    LogReg = SWLogisticRegression::ComputeModel( dataTrain );
                    AccuracyNow += SWLogisticRegression::TestModel( LogReg, dataTest );
                }
                AccuracyNow /= nk;
                LRAccuracy += AccuracyNow;
            }
            LRAccuracy /= 10;

            LRAccuracyUpToDate = true;

            //Using all data to update model
            LogReg = SWLogisticRegression::ComputeModel( data );
            diffParams.NbAttemptsUsedToCompute = data->DepVar.Num();
        }
        else
        {
            data = data->shuffle();
            LogReg = SWLogisticRegression::ComputeModel( data );
            diffParams.NbAttemptsUsedToCompute = data->DepVar.Num();
        }

        if ( LRAccuracy < LRMinimalAccuracy )
        {
            //Debug.Log( "LogReg accuracy is under " + LRMinimalAccuracy + ", not using LogReg" );
            diffParams.LogRegReady = false;
            diffParams.LogRegError = ESWDDALogRegError::ACCURACY_TOO_LOW;
        }

        if ( !LogReg->isUsable() )
        {
            LRAccuracy = 0;
            diffParams.LogRegError = ESWDDALogRegError::NEWTON_RAPHSON_ERROR;
        }
        else if ( diffParams.LogRegReady )
        {
            //Verifying if LogReg is ok : must be able to work in both ways
            auto errorSum = 0.f;
            auto diffTest = 0.1f;
            TArray<float> pars;
            TArray<float> parsForAllDiff;
            FString res;
            for (auto index = 0; index < 8; ++index)
            {
                pars[0] = LogReg->InvPredict(diffTest, pars, 0); //on regarde que la première variable.
                parsForAllDiff[index] = pars[0];
                res = "D = " + FString::SanitizeFloat(diffTest) + " par = " + FString::SanitizeFloat(pars[0]);
                errorSum += FMath::Abs(diffTest - LogReg->Predict(pars)); //On passe dans les deux sens on doit avoir pareil
                res += " res = " + FString::SanitizeFloat(LogReg->Predict(pars)) + "\n";
                diffTest += 0.1;
                //Debug.Log(res);
            }
            
            if (errorSum > 1 || FMath::IsNaN( errorSum ))
            {
                //Debug.Log("Model is not solid, error = " + errorSum);
                LRAccuracy = 0;
                if (errorSum > 1)
                    diffParams.LogRegError = ESWDDALogRegError::SUM_ERROR_TOO_HIGH;
                if (FMath::IsNaN( errorSum ))
                    diffParams.LogRegError = ESWDDALogRegError::SUM_ERROR_IS_NAN;
            }


            //Verifying if LogReg is ok : sd of diff predictions in all theta range must not be 0
            float mean = 0;
            for (auto index = 0; index < 8; ++index)
                mean += parsForAllDiff[index];
            mean /= 8;
            float sd = 0;
            for (auto index = 0; index < 8; ++index)
                sd += (parsForAllDiff[index] - mean) * (parsForAllDiff[index] - mean);
            sd = FMath::Sqrt(sd);

            //Debug.Log("Model parameter estimation sd = " + sd);

            if (sd < 0.05 || FMath::IsNaN( sd ))
            {
                //Debug.Log("Model parameter estimation is always the same : sd=" + sd);
                LRAccuracy = 0;

                if (sd < 0.05)
                    diffParams.LogRegError = ESWDDALogRegError::SD_PRED_TOO_LOW;
                if (FMath::IsNaN( sd ))
                    diffParams.LogRegError = ESWDDALogRegError::SD_PRED_IS_NAN;
            }
        }
    }

    //Saving params
        diffParams.TargetDiff = targetDifficulty;
        diffParams.LRAccuracy = LRAccuracy;

        //Determining theta

        //If we want pmdelta or we want log reg but it's not available
        if ((Algorithm == ESWDDAAlgorithm::DDA_LOGREG && !diffParams.LogRegReady) ||
             Algorithm == ESWDDAAlgorithm::DDA_PMDELTA)
        {
            auto delta = PMWonLastTime ? PMDeltaValue : -PMDeltaValue;
            delta *= FMath::RandRange(PMDeltaExploMin, PMDeltaExploMax);
            diffParams.Theta = PMLastTheta + delta;
            diffParams.AlgorithmActuallyUsed = ESWDDAAlgorithm::DDA_PMDELTA;

            //If regression is okay, we can tell the difficulty for this theta
            if (diffParams.LogRegReady)
            {
                TArray<float> pars;
                pars.Add( diffParams.Theta);
                diffParams.TargetDiff = 1.0 - LogReg->Predict(pars);
                diffParams.TargetDiffWithExplo = diffParams.TargetDiff;
            }
            else //Otherwise we just can tell we aim for 0.5
            {
                diffParams.TargetDiffWithExplo = 0.5;
                diffParams.TargetDiff = 0.5;
            }
        }

        //if we want log reg and it's available
        if (Algorithm == ESWDDAAlgorithm::DDA_LOGREG && diffParams.LogRegReady)
        {
            diffParams.TargetDiffWithExplo = targetDifficulty + FMath::RandRange(-LRExplo, LRExplo);
            diffParams.TargetDiffWithExplo = FMath::Min(1.0f, FMath::Max(0.f, static_cast< float >( diffParams.TargetDiffWithExplo )));
            diffParams.Theta = LogReg->InvPredict(1.0f - diffParams.TargetDiffWithExplo);
            diffParams.AlgorithmActuallyUsed = ESWDDAAlgorithm::DDA_LOGREG;
        }

        //if we want random log reg and it's available
        if (Algorithm == ESWDDAAlgorithm::DDA_RANDOM_LOGREG && diffParams.LogRegReady)
        {
            diffParams.TargetDiff = FMath::RandRange(0.0f, 1.0f);
            diffParams.TargetDiffWithExplo = diffParams.TargetDiff; //Pas d'explo on est en random
            diffParams.Theta = LogReg->InvPredict(1.0f - diffParams.TargetDiffWithExplo);
            diffParams.AlgorithmActuallyUsed = ESWDDAAlgorithm::DDA_RANDOM_LOGREG;
        }

        //If we want random
        if (Algorithm == ESWDDAAlgorithm::DDA_RANDOM_THETA || (Algorithm == ESWDDAAlgorithm::DDA_RANDOM_LOGREG && !diffParams.LogRegReady))
        {
            diffParams.Theta = FMath::RandRange(0.0f, 1.0f);
            diffParams.AlgorithmActuallyUsed = ESWDDAAlgorithm::DDA_RANDOM_THETA;

            //If regression is okay, we can tell the difficulty for this theta
            if (diffParams.LogRegReady)
            {
                TArray<float> pars;
                pars.Add(diffParams.Theta);
                diffParams.TargetDiff = 1.0 - LogReg->Predict(pars);
                diffParams.TargetDiffWithExplo = diffParams.TargetDiff;
            }
            else //Otherwise, we don't know, let's put a negative value
            {
                diffParams.TargetDiffWithExplo = -1;
                diffParams.TargetDiff = -1;
            }
        }

        //Save betas if we have some
        if (LogReg != nullptr && LogReg->Betas.Num() > 0)
        {
            diffParams.Betas.Reset(LogReg->Betas.Num());
            for (auto index = 0; index < LogReg->Betas.Num(); ++index)
                diffParams.Betas.Add(LogReg->Betas[index]);
        }

        //Clamp 01 float. Super inportant pour éviter les infinis
        diffParams.Theta = diffParams.Theta > 1.0 ? 1.0 : diffParams.Theta;
        diffParams.Theta = diffParams.Theta < 0.0 ? 0.0 : diffParams.Theta;

        return diffParams;
 }

bool USWDDAModel::checkDataAgainst( TArray< USWDDAAttempt * > & attempts ) const
{
    auto attemptsSaved = DataManager->getAttempts( PlayerId, ChallengeId, LRNbLastAttemptsToConsider );

    auto isSame = true;
    auto nbCheck = 0;
    for ( auto index = 0; index < attempts.Num(); ++index )
    {
        if ( index >= attempts.Num() - LRNbLastAttemptsToConsider )
        {
            if ( !attempts[ index ]->IsSame( attemptsSaved[ index - ( attempts.Num() - LRNbLastAttemptsToConsider ) ] ) )
            {
                //Debug.LogError( "Attempt " + index + " is corrupted" );
                isSame = false;
            }
            nbCheck++;
        }
    }

    //if ( isSame )
    //Debug.Log( "Data is ok, checked " + nbCheck + " attempts (cache size)" );

    return isSame;
}
