#include "SWDDADataManager_LocalCSV.h"

#include "SWCacheData.h"
#include "SWDDAAttempt.h"

#include <Misc/FileHelper.h>

USWDDADataManager_LocalCSV::USWDDADataManager_LocalCSV()
{
    FileDataName = "data.csv";
}

void USWDDADataManager_LocalCSV::addAttempt( const FString playerId, const FString challengeId, USWDDAAttempt * attempt )
{
    //On va stocker les donnees en cache
    auto * cache = findCache(playerId, challengeId);
    if (cache != nullptr) //Sinon il sera créé au load, ou on a la taille limite pour dimensionner le cache
        cache->addAttempt(attempt);

    //On sauve
    const auto csvFile = FPaths::ProjectDir() + playerId + "_" + challengeId + FileDataName;
    FString content;
            
        for (auto i = 0; i < attempt->Thetas.Num(); i++)
        {
            content.Append(FString::SanitizeFloat(attempt->Thetas[i]));
            content.Append(";");
        }
        content.Append(FString::SanitizeFloat(attempt->Result));
        content.Append("\n");
            
    FFileHelper::SaveStringToFile(content, *csvFile, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), EFileWrite::FILEWRITE_Append);
}

TArray<USWDDAAttempt *> USWDDADataManager_LocalCSV::getAttempts( const FString playerId, const FString challengeId, const int nbLastAttempts )
{
    //On va stocker les donnees en cache
    auto * cache = findCache(playerId, challengeId);

    if(cache != nullptr)
    {
        if(cache->SizeLimit == nbLastAttempts)
        {
            //On a deja les données en cache et c'est la bonne taille, on les retourne
            return cache->Attempts;
        }
        else
        {
            //Pas la meme taille, on va recharger tout le fichier
            GEngine->AddOnScreenDebugMessage(-1, 1000.f, FColor::Red, "Warning !! you need to always retrieve the same number of attempts for performance reasons. If cache size changes, file need to be loaded again.");
            deleteCache(playerId, challengeId);
            cache = nullptr;
        }

    }
    
    //On a pas les données en cache, on crée un nouveau cache
    cache = createCache(playerId, challengeId, nbLastAttempts);

    const auto csvFile = FPaths::ProjectDir() + playerId + "_" + challengeId + FileDataName;

    TArray<FString> FileData;
    FFileHelper::LoadFileToStringArray( FileData, *csvFile );

    //Counting number of lines and variables
    TArray<FString> tokens;

    //For first line, test if headers
    auto line = FileData[0].TrimStartAndEnd();
    line.ParseIntoArray( tokens, TEXT(";"), false);
    const auto nbVars = tokens.Num() - 1; //Removing dependant variable

    const auto bHeaders = FCString::Atof(*tokens[0]) == 0;

    const auto ct = FileData.Num() - (bHeaders ? 1 : 0); // Nombre de lignes (sans les headers)

    for (auto row = 0; row < FileData.Num(); ++row)
    {
        if (row >= (ct - nbLastAttempts))
        {
            line = FileData[row].TrimStartAndEnd();
            line.ParseIntoArray( tokens, TEXT(";"), false);
            auto * attempt = NewObject<USWDDAAttempt>();
            attempt->Thetas.Reserve(nbVars);
            for (auto index = 0; index < nbVars; index++)
            {
                attempt->Thetas.Add(FCString::Atof( *tokens[index] ));
            }
            attempt->Result = FCString::Atof( *tokens[tokens.Num() - 1] );
            cache->addAttempt(attempt);
        }                
    }

    return cache->Attempts;
}

USWCacheData * USWDDADataManager_LocalCSV::findCache( const FString playerId, const FString challengeId )
{
    USWCacheData * cache = nullptr;
    for (auto * lcache : Caches)
    {
        if (lcache->PlayerId == playerId && lcache->ChallengeId == challengeId)
        {
            cache = lcache;
        }
    }

    return cache;
}

USWCacheData * USWDDADataManager_LocalCSV::createCache( const FString playerId, const FString challengeId, const int sizeLimit )
{
    auto * cache = findCache(playerId, challengeId);
      
    if (cache == nullptr)
    {
        cache = NewObject<USWCacheData>();
        cache->Init(playerId, challengeId, sizeLimit);
        Caches.Add(cache);
    }

    return cache;
}

void USWDDADataManager_LocalCSV::deleteCache( FString playerId, FString challengeId )
{
    Caches.RemoveAll([playerId, challengeId](USWCacheData * item) {return item->PlayerId == playerId && item->ChallengeId == challengeId;});
}
