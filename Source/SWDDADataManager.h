#pragma once

#include <CoreMinimal.h>

#include "SWDDaDataManager.generated.h"

class USWDDAAttempt;

UCLASS( Abstract )
class SWARMS_API USWDDADataManager : public UObject
{
    GENERATED_BODY()

public:
    //Save all these new attempts for this player and this challenge
    virtual void addAttempt( FString playerId, FString challengeId, USWDDAAttempt * attempt )
    {}

    //Get nbLastAttempts of this player for this challenge
    virtual TArray< USWDDAAttempt * > getAttempts( FString playerId, FString challengeId, int nbLastAttempts )
    {
        return TArray< USWDDAAttempt * >();
    }
};