#include "SWCacheData.h"

void USWCacheData::Init( const FString playerId, const FString challengeId, const int sizeLimit )
{
    PlayerId = playerId;
    ChallengeId = challengeId;
    SizeLimit = sizeLimit;
}

void USWCacheData::addAttempt( USWDDAAttempt * attempt )
{
    Attempts.Add( attempt );

    if ( Attempts.Num() > SizeLimit )
        Attempts.RemoveAt( 0 );
}
