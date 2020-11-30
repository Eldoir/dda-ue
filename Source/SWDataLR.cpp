#include "SWDataLR.h"

#include "SWLogisticRegression.h"

#include <Misc/FileHelper.h>

USWDataLR::USWDataLR()
{
    IndepVar = SWLogisticRegression::MatrixCreate(0, 0);
    DepVar.Empty();
}

USWDataLR * USWDataLR::shuffle()
{
    auto * part = NewObject< USWDataLR >();

    if ( DepVar.Num() == 0 )
        return part;

    const auto nbRows = DepVar.Num();
    const auto nbVars = IndepVar[ 0 ].Num();

    part->IndepVar.Reset( nbRows );
    for (auto index = 0; index < nbRows; ++index)
    {
        part->IndepVar.Add(TArray<float>());
    }
    part->DepVar = SWLogisticRegression::VectorCreate( nbRows );

    auto row = 0;
    for ( auto & vars : IndepVar )
    {
        //On tire une ligne a remplir au hasard
        const auto rowRand = FMath::RandRange(0, TNumericLimits<int>::Max()) % nbRows;
        //sens dans lequel on cherche une case vide
        const auto sens = FMath::RandRange(0.f, 1.f) > 0.5 ? -1 : 1;

        //On cherche une case vide
        auto nextRow = -1;
        if ( sens > 0 )
        {
            for ( auto index = 0; index < nbRows; ++index )
            {
                const auto rowTest = ( rowRand + index ) % nbRows;
                if ( part->IndepVar[ rowTest ].Num() == 0 )
                    nextRow = rowTest;
            }
        }
        else
        {
            for ( auto index = 0; index < nbRows; ++index )
            {
                auto rowTest = ( rowRand - index );
                if ( rowTest < 0 )
                    rowTest += nbRows;
                if ( part->IndepVar[ rowTest ].Num() == 0 )
                    nextRow = rowTest;
            }
        }

        if (nextRow < 0)
            throw new std::exception("Pas trouvé de case vide, algo de shuffle marche pas");

        part->IndepVar[nextRow].Reset(nbVars);
                
        for (auto index = 0; index < vars.Num(); ++index)
        {
            part->IndepVar[nextRow].Add(vars[index]);
        }
        part->DepVar[nextRow] = DepVar[row];

        ++row;
    }
    
    return part;
}

void USWDataLR::split( const int pcentStartExtract, const int pcentEndExtract, USWDataLR * partOut, USWDataLR * partIn )
{
    partOut = NewObject<USWDataLR>();
    partIn = NewObject<USWDataLR>();

    if (DepVar.Num() == 0)
        return;

    const auto nbLignes = DepVar.Num();
    const auto nbVars = IndepVar[0].Num();

    const auto iStart = (nbLignes * pcentStartExtract) / 100;
    const auto iEnd = (nbLignes * pcentEndExtract) / 100;
    const auto nbRowsIn = iEnd - iStart;
    const auto nbRowsOut = DepVar.Num() - nbRowsIn;

    partIn->IndepVar = SWLogisticRegression::MatrixCreate(nbRowsIn, nbVars);
    partIn->DepVar.Reset(nbRowsIn);

    partOut->IndepVar = SWLogisticRegression::MatrixCreate(nbRowsOut, nbVars);
    partOut->DepVar.Reset(nbRowsOut);

    auto row = 0;
    auto rowIn = 0;
    auto rowOut = 0;
    for (auto & vars : IndepVar)
    {
        //out of section
        if (row < iStart || row >= iEnd)
        {
            for (auto index = 0; index < vars.Num(); ++index)
                partOut->IndepVar[rowOut][index] = vars[index];
            partOut->DepVar.Add(DepVar[row]);
            ++rowOut;
        }

        //in section
        if (row >= iStart && row < iEnd)
        {
            for (auto index = 0; index < vars.Num(); ++index)
                partIn->IndepVar[rowIn][index] = vars[index];
            partIn->DepVar.Add(DepVar[row]);
            ++rowIn;
        }

        ++row;
    }

    //Console.WriteLine("Partion split: partIn=" + rowIn + " partOut:" + rowOut);
}

USWDataLR * USWDataLR::getLastNRows( const int nbRows )
{
    auto * part = NewObject<USWDataLR>();
    if(DepVar.Num() > 0)
    {
        const auto nbRowsTake = FMath::Min(nbRows, DepVar.Num());
        const auto nbVars = IndepVar[0].Num();
        const auto iStart = DepVar.Num() - nbRowsTake;

        part->IndepVar = SWLogisticRegression::MatrixCreate(nbRowsTake, nbVars);
        part->DepVar = SWLogisticRegression::VectorCreate(nbRowsTake);

        auto row = 0;
        auto rowLoad = 0;
        for (auto & vars : IndepVar)
        {
            //out of section
            if (row >= iStart)
            {
                for (auto index = 0; index < vars.Num(); ++index)
                    part->IndepVar[rowLoad][index] = vars[index];
                part->DepVar[rowLoad] = DepVar[row];
                ++rowLoad;
            }
            ++row;
        }
    }

    return part;
}

void USWDataLR::LoadDataFromList( TArray<TArray<float> >  & indepVars, TArray<float>  & depVars )
{
    if(indepVars.Num() == 0)
    {
        IndepVar = SWLogisticRegression::MatrixCreate(0, 0);
        DepVar.Empty();
        return;
    }

    const auto nbRows = depVars.Num();
    IndepVar = SWLogisticRegression::MatrixCreate(nbRows, indepVars[0].Num() + 1);
    DepVar.Reset(nbRows);

    auto row = 0;
    for (auto & vars : indepVars)
    {
        IndepVar[row][0] = 1;
        for (auto index = 0; index < vars.Num(); ++index)
            IndepVar[row][index+1] = vars[index];
        DepVar.Add(depVars[row]);
        ++row;
    }
}

void USWDataLR::LoadDataFromCsv( const FString csvFile )
{
    if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*csvFile))
    {
        GEngine->AddOnScreenDebugMessage(-1, 1000.f, FColor::Red, TEXT("Could not Find File"));
        return;
    }

    TArray<FString> FileData;
    FFileHelper::LoadFileToStringArray( FileData, *csvFile );
    
    //On compte le nombre de lignes et de variables
    TArray<FString> tokens;

    //Si lgne 1, on test si headers
    auto line = FileData[0].TrimStartAndEnd();
    line.ParseIntoArray( tokens, TEXT(";"), false);
    const auto nbVars = tokens.Num() - 1; //On compte aussi la variable dépendante

    const auto bHeaders = FCString::Atof(*tokens[0]) == 0;

    const auto ct = FileData.Num() - (bHeaders ? 1 : 0); // Nombre de lignes (sans les headers)

    //On parse le fichier pour charger les datas
    IndepVar = SWLogisticRegression::MatrixCreate(ct, nbVars + 1);
    DepVar = SWLogisticRegression::VectorCreate(ct);
    FFileHelper::LoadFileToStringArray( FileData, *csvFile );

    for (auto row = 0; row < FileData.Num(); ++row)
    {
        line = FileData[row].TrimStartAndEnd();
        line.ParseIntoArray( tokens, TEXT(";"), false);
        IndepVar[row][0] = 1.f;
        for (auto index = 0; index < nbVars; ++index)
        {
            IndepVar[row][index + 1] = FCString::Atof( *tokens[index] );
        }
        DepVar[row] = FCString::Atof(*tokens[tokens.Num() - 1]);
    }
}

void USWDataLR::saveDataToCsv( const FString csvFile )
{
    FString content;

    auto row = 0;
    for (auto & vars : IndepVar)
    {
        for (auto index = 1; index < vars.Num(); ++index)
        {
            content.Append( FString::SanitizeFloat( vars[index] ) );
            content.Append( ";" );
        }
        content.Append( FString::SanitizeFloat( DepVar[row] ) );
        content.Append("\n");
        ++row;
    }

    FFileHelper::SaveStringToFile( content, *csvFile );
}
