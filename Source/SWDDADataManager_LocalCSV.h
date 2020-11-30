#pragma once

#include "SWDDADataManager.h"

#include <CoreMinimal.h>

#include "SWDDADataManager_LocalCSV.generated.h"

class USWCacheData;

UCLASS(BlueprintType)
class USWDDADataManager_LocalCSV : public USWDDADataManager
{
    GENERATED_BODY()

public:
    USWDDADataManager_LocalCSV();

    //Save all these new attempts for this player and this challenge
    void addAttempt( FString playerId, FString challengeId, USWDDAAttempt * attempt ) override;
    //Get nbLastAttempts of this player for this challenge
    TArray< USWDDAAttempt * > getAttempts( FString playerId, FString challengeId, int nbLastAttempts ) override;

private:
    USWCacheData * findCache( FString playerId, FString challengeId );
    USWCacheData * createCache( FString playerId, FString challengeId, int sizeLimit );
    void deleteCache( FString playerId, FString challengeId );

    FString FileDataName;
    TArray< USWCacheData * > Caches;
};