#Custom files######################## 
from recorder import * 
from speaker_identification import * 
##################################### 
TRAIN   = "train" 
IDENTIF = "identification"  
TEST  = "test"
##########
MODE = IDENTIF
###########
if __name__ == "__main__":    
    if MODE == TRAIN:     
        # Initialize speaker identification system         
        speaker_id = SpeakerIdentification(n_components=256)   
        # Train the system         
        data_dir = "../audio"         
        speaker_id.train(data_dir)
        # Save the trained models         
        speaker_id.save_models("speaker_models.pkl")     
        
    elif  MODE == IDENTIF :
        #speaker_id = SpeakerIdentification(n_components=256)          
        # for loading instead of training again #         
        speaker_id = SpeakerIdentification.load_models("speaker_models.pkl")
        test_audio = "./test_data/moetaz.wav"
        #test_audio = record_voice() 
        identified_speaker, scores = speaker_id.identify_speaker(test_audio)           
        #print("\nScores for each speaker:") 
        tempscore = 0    
        for speaker, score in scores.items():                
            print(f"{speaker}: {score:.2f}")
            tempscore += abs(score)
        if tempscore > 200 :
            identified_speaker = "Unknown"         
        print(f"\nIdentified speaker: {identified_speaker}")

    # if MODE == TEST:
    #     speaker_id = SpeakerIdentification.load_models("speaker_models.pkl")

    #     audio_path = "../audio/omar"
    #     count = 0
    #     for audio_file in os.listdir(audio_path):
    #         print(audio_file)
    #         identified_speaker, scores = speaker_id.identify_speaker(f"{audio_path}/{audio_file}")
    #         if (identified_speaker == "omar"):
    #             count +=1
    #     print(f"Precision: {(count/len(os.listdir(audio_path)))*100}%")

    