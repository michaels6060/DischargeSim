import argparse
import re, random, time, json, os
import openai
import csv
from vllm import LLM, SamplingParams
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Dict
import datasets
import transformers

class Medicine(BaseModel):
    name: str = Field(..., alias="Name")
    dosage: str = Field(..., alias="Dosage")
    reason: str = Field(..., alias="Reason")
    how_to_take: str = Field(..., alias="How to take")

class Medicines(BaseModel):
    Morning: Dict[str, Medicine]
    Afternoon: Dict[str, Medicine]
    Evening: Dict[str, Medicine]

class AHRQ(BaseModel):
    Diagnosis: str
    Exercises: str
    Food: str
    activites_or_food_to_avoid: str = Field(..., alias="Activites or Food to Avoid")
    Medicines: Medicines

class DischargeDiagnosis(BaseModel):
    primary: str = Field(..., alias="PRIMARY")
    secondary: str = Field(..., alias="SECONDARY")

class DischargeCondition(BaseModel):
    mental_status: str = Field(..., alias="Mental Status")
    level_of_consciousness: str = Field(..., alias="Level of Consciousness")
    activity_status: str = Field(..., alias="Activity Status")

class DischargeInfo(BaseModel):
    discharge_medications: str = Field(..., alias="Discharge Medications")
    discharge_diagnosis: DischargeDiagnosis = Field(..., alias="Discharge Diagnosis")
    discharge_condition: DischargeCondition = Field(..., alias="Discharge Condition")
    discharge_instructions: str = Field(..., alias="Discharge Instructions")
    followup_instructions: str = Field(..., alias="Followup Instructions")

def query_model(model_str, prompt, system_prompt, tries=30, timeout=6000.0, scene=None, max_prompt_len=2**14, clip_prompt=False):
    for _ in range(tries):
        if clip_prompt: prompt = prompt[:max_prompt_len]
        answer = ""
        try:
            str_to_model = {"gpt3.5" : "gpt-3.5-turbo", "gpt-4o-mini": "gpt-4o-mini", "gpt4o" : "gpt-4o", "gpt-5" : "gpt-5", "gpt-5-mini" : "gpt-5-mini", "gpt-5-nano" : "gpt-5-nano"}
            if model_str in str_to_model:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                if model_str in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                    response = openai.chat.completions.create(
                        model=str_to_model[model_str],
                        messages=messages,
                        reasoning_effort="minimal",
                        max_completion_tokens=2000,
                    )
                else:
                    response = openai.chat.completions.create(
                        model=str_to_model[model_str],
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer= response.choices[0].message.content
                answer = re.sub("\s+", " ", answer)

            elif model_str == "gpt-4.1-mini" or model_str == "gpt-4.1-nano":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response = openai.chat.completions.create(
                    model=model_str,
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer= response.choices[0].message.content
                answer = re.sub("\s+", " ", answer)
            elif model_str == "gpt-4.1":
                client = OpenAI(
                    base_url="https://xiaoai.plus/v1",
                    api_key="sk-tbT6ioTp4949r11ENNKu8ZOsHt8sF4kt4ypqE1EXQYP8C8SI",
                    http_client=httpx.Client(
                        base_url="https://xiaoai.plus/v1",
                        follow_redirects=True,
                    ),
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                #response = client.chat.completions.create(
                response = client.chat.completions.create(
                    model=model_str,
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer= response.choices[0].message.content
                answer = re.sub("\s+", " ", answer)


         
            elif model_str == "mistral":
                client = openai.OpenAI(
                    base_url=f"http://localhost:{VLLM_PORT}/v1",
                    api_key="-",
                )
                messages = [
                    {"role": "user", "content": system_prompt + "\n" + prompt}]
                response = client.chat.completions.create(
                        model=LOCAL_MODEL,
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer= response.choices[0].message.content
                answer = re.sub("\s+", " ", answer)
            else:
                client = openai.OpenAI(
                    base_url=f"http://localhost:{VLLM_PORT}/v1",
                    api_key="-",
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                
                response = ""
                if model_str == "qwen3":
                    response = client.chat.completions.create(
                            model=LOCAL_MODEL,
                            messages=messages,
                            temperature=0.05,
                            max_tokens=200,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                        )
                else:
                    response = client.chat.completions.create(
                            model=LOCAL_MODEL,
                            messages=messages,
                            temperature=0.05,
                            max_tokens=200,
                        )
                answer= response.choices[0].message.content
                answer = re.sub("\s+", " ", answer)

            return answer
        
        except Exception as e:
            print(e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")



class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        # currently hard coded. medqa in json format
        with open("data/medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

        self.original = scenario_dict.get("original", "")
    
    def original_note(self) -> dict:
        return self.original

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis

class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        # currently hard coded
        # opens OSCE formatted json files for each case, can be modified to take in only mimiciv notes
        with open("data/osce-format.jsonl", "r") as f:
            main_scenarios = [json.loads(line) for line in f]
        # handles original notes
        with open("data/original-notes.txt", "r") as f2:
            original_notes = [line.strip().strip('"') for line in f2 if line.strip()]

        for i in range(min(len(main_scenarios), len(original_notes))):
            original_text = original_notes[i]
            if "Discharge Condition" in original_text:
                original_text = original_text.split("Discharge Condition")[0]
            main_scenarios[i]["original"] = original_text
        self.scenario_strs = main_scenarios
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        self.disease = ""
        self.symptoms = ""
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.reset()

        # currently unused 
        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]
        import random
        self.education = random.randint(1, 4)

    # currently unused
    def generate_bias(self) -> str:
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def gen_bias2(self):
        return self.bias_present
    
    def inference_patient(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of your dialogue: \"" + self.agent_hist + "\" \n Here was the doctor response: \"" + question + "\" \n Continue the dialogue with only your response as a patient\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            #bias_prompt = self.generate_bias()
            if self.bias_present == "education1":
                bias_prompt = "You have not completed high school and have a low level of education. This affects you understanding and how you interact with your doctor."
            elif self.bias_present == "education2":
                bias_prompt = "You have completed high school and your GED. This affects you understanding and how you interact with your doctor."
            elif self.bias_present == "education3":
                bias_prompt = "You have completed your bachelor's degree. This affects you understanding and how you interact with your doctor."
            elif self.bias_present == "HL1":
                bias_prompt = "You have a low health literacy, you are not very sure of what is healthy or how to properly take care of yourself. This affects you understanding and how you interact with your doctor."
            elif self.bias_present == "HL2":
                bias_prompt = "You have a high health literacy, you are aware of what is healthy and know more about how to care for yourself. This affects you understanding and how you interact with your doctor."
            elif self.bias_present == "anxious":
                bias_prompt = "You have an anxious response style. This affects how you interact with your doctor."
            elif self.bias_present == "deflective":
                bias_prompt = "You have an deflective response style. This affects how you interact with your doctor."
            elif self.bias_present == "neutral":
                bias_prompt = "You have an neutral response style. This affects how you interact with your doctor."
            else:
                raise Exception()
        
        base = """You are a patient in a hospital who only responds in the form of dialogue. You are preparing to be discharged. You will engage in a dialogue with your doctor. Your response should be brief (1-3 sentences)."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n You were given this diagnosis {}. You do not have a full understanding of what this diangosis means.".format(self.symptoms, self.scenario.diagnosis)
        return base + symptoms + bias_prompt
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"



class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=50, bias_present=None) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.reset()
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

        self.discharge_infs = 0
        self.log = ""
        self.stage_infs = 0
        self.max_stage_infs = 5
        self.stage = 0
        self.NUMBER_OF_STAGES = 6

    def generate_bias(self) -> str:
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_doctor(self, question) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        answer = query_model(self.backend, "\nHere is a history of your dialogue: \"" + self.agent_hist + "\" \n Here was the patient response: \"" + question + "\" \n Continue the dialogue with only your response as a doctor\nDoctor: ", self.system_prompt(), scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        answer = answer.strip('\"')
        if len(question) == 0:
            self.log += "Doctor: " + answer + "\n"
        else:
            self.log += "Patient: " + question + "\n" + "Doctor: " + answer + "\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()

        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are in the process of discharging a patient after they finished visiting the hospital. "
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Below is the original clinical note: {} ".format(self.presentation, self.scenario.original_note())

        self.stage_infs = (self.stage_infs + 1) % self.max_stage_infs
        if self.stage_infs % self.max_stage_infs == 0:
            self.stage += 1

        stage_prompt =""
        stage_suffix = "You should cover only the top one or two most important points for your current goal. If your conversation already covered your goal you can continue on. "
        if self.stage == 0:
            self.max_stage_infs = 5
            stage_prompt = "Your current goal is to inform and help the patient understand their diagnosis. " + stage_suffix
        elif self.stage == 1:
            stage_prompt = "Your current goal is to inform and help the patient understand what tests and treatments they received at the hospital. " + stage_suffix
        elif self.stage == 2:
            stage_prompt = "Your current goal is to inform and help the patient understand any important signs that indicate they should return to the hospital. " + stage_suffix
        elif self.stage == 3:
            self.max_stage_infs = 10
            stage_prompt = "Your current goal is to inform and help the patient understand every single one of the medications that they should take after being discharged, including information such as dosages and frequency to take. "
        elif self.stage == 4:
            self.max_stage_infs = 5
            stage_prompt = "Your current goal is to inform and help the patient understand their postdischarge treatment, such as any specific activities to do or avoid. " + stage_suffix + "If there are none you can skip this. "
        elif self.stage == 5:
            stage_prompt = "Your current goal is to inform and help the patient understand if or when they should follow-up. " + stage_suffix + "If it is not necessary you can skip this. "
        else:
            termination = "Finish your dialogue. Your next response must include \"FINISHED SIMULATION\" at the end."
            return base + termination

        next_stage = "If you feel like you have finished going through all the important points for your current goal, add \"NEXT STAGE\" to the end of your response."

        if self.infs == self.MAX_INFS-1:
            termination = "Finish your dialogue. Your next response must include \"FINISHED SIMULATION\" at the end."
            return base + termination

        return base + bias_prompt + stage_prompt + presentation + next_stage

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()

    def discharge_system_prompt(self, plan) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are providing discharge care. Your goal is to confirm that the patient understands the discharge instructions and diagnosis, including their aftervisit plan, medication regimen, follow-up appointments, and any activity restrictions. Your response should be brief (1-3 sentences). You are only allowed to ask {} questions total. You have asked {} questions so far.".format(5, self.discharge_infs) 
        presentation = "\n\nBelow is discharge plan. {}. \n\n Remember, you must discover if the patinet is confused on any parts of their plan by asking them questions.".format(plan)
        return base + bias_prompt + presentation


def main(api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm, num_scenarios, dataset, total_inferences, output_file):
    openai.api_key = api_key
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))
    for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
        pi_dialogue = str()
        scenario =  scenario_loader.get_scenario(id=_scenario_id)
        patient_agent = PatientAgent(
            scenario=scenario, 
            bias_present=patient_bias,
            backend_str=patient_llm)
        doctor_agent = DoctorAgent(
            scenario=scenario, 
            bias_present=doctor_bias,
            backend_str=doctor_llm,
            max_infs=total_inferences)

        doctor_dialogue = ""
        for _inf_id in range(total_inferences):
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue)
            print("Doctor [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), doctor_dialogue)
            if "NEXT STAGE" in doctor_dialogue:
                doctor_agent.stage += 1
                doctor_agent.stage_infs = 0
                    
            if "FINISHED SIMULATION" in doctor_dialogue or _inf_id == total_inferences - 1:
                example = """
                Here is an example structure
                {
                    "Discharge Medications": "
                        1. Docusate Sodium 100 mg Capsule Sig: One (1) Capsule PO BID (2 
                        times a day) as needed for constipation.  
                        2. Senna 8.6 mg Tablet Sig: One (1) Tablet PO BID (2 times a 
                        day) as needed for constipation.
                        Disp:*60 Tablet(s)* Refills:*2*
                        3. Ascorbic Acid ___ mg Tablet Sig: One (1) Tablet PO BID (2 
                        times a day).  
                    ",
                    "Discharge Diagnosis": {
                        "PRIMARY": "
                            - Acute muscular hematoma, right flank
                            - Hemophilia, factor VIII deficiency
                        ",
                        "SECONDARY": ""
                    },
                    "Discharge Condition":{
                        "Mental Status": "Clear and coherent.",
                        "Level of Consciousness": "Alert and interactive.",
                        "Activity Status": "Ambulatory - Independent."
                    },
                    "Discharge Instructions": "
                        Mr. ___,

                        It was our pleasure caring for you at ___ 
                        ___. You were admitted with bruising on your right 
                        side and low blood counts after a snowboarding fall. With your 
                        history of hemophilia, it was important to evaluate internal 
                        bleeding which did show a right muscular flank blood collection. 
                        Your facotr VIII level was 103 and you received IV DDAVP under 
                        our care. Your blood counts were stable to improved on the day 
                        of admission. 

                        It is important that you not participate in any dangerous 
                        activities given your recent bleed and your hemophilia. Bleeding 
                        in hemophiliacs has more potential to be life-threatening.

                        Please get your blood counts checked at ___ site on either 
                        ___ or ___. Follow up with your 
                        regular doctor early next week.

                        Best wishes,
                        Your ___ Care Team
                    ",
                    "Followup Instructions": "___"
                }
                """

                ahrqexample = """
                Here is an example of the structure:
                {
                    "AHRQ": {
                        "Diagnosis": "Diagnosis",
                        "Exercises" : "None", 
                        "Food": "None", 
                        "Activites or Food to Avoid": "None", 
                        "Medicines" : {
                        "Morning": {
                            "Medicine 1" : {
                            "Name": "Name",
                            "Dosage" : "Dosage",
                            "Reason" : "Reason",
                            "How to take" : "Directions to take medicine"
                            }
                        },
                        "Afternoon": {
                            "Medicine 2" : {
                            "Name": "Name",
                            "Dosage" : "Dosage",
                            "Reason" : "Reason",
                            "How to take" : "Directions to take medicine"
                            }
                        },
                        "Evening": {
                            "Medicine 3" : {
                            "Name": "Name",
                            "Dosage" : "Dosage",
                            "Reason" : "Reason",
                            "How to take" : "Directions to take medicine"
                            }
                        },
                    }
                }
                """


                str_to_model = {"gpt3.5" : "gpt-3.5-turbo", "gpt-4o-mini": "gpt-4o-mini", "gpt4o" : "gpt-4o", "gpt-4.1-mini" : "gpt-4.1-mini", "gpt-4.1-nano" : "gpt-4.1-nano", "gpt-4.1" : "gpt-4.1", "gpt-5" : "gpt-5", "gpt-5-mini" : "gpt-5-mini", "gpt-5-nano" : "gpt-5-nano"}
                ahrq_output = ""
                summary_out = ""
                if doctor_llm in str_to_model:
                    m2 = [
                        {"role": "system", "content": doctor_agent.system_prompt()},
                        {"role": "user", "content": f"\nHere is a history of your dialogue: " + doctor_agent.agent_hist + f"\n Generate an after visit plan based on your interactions according to the following example plan in a json format for each subheading given in the following example: {ahrqexample} .\n\n Be sure to list all mediciations."}]
                    if doctor_llm in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                        ahrq_output = openai.chat.completions.create(
                            model=str_to_model[doctor_llm],
                            messages=m2,
                            response_format={ "type": "json_object" }
                        )
                    else:
                        ahrq_output = openai.chat.completions.create(
                            model=str_to_model[doctor_llm],
                            messages=m2,
                            response_format={ "type": "json_object" }
                        )
                
                    ahrq_output = ahrq_output.choices[0].message.content
                    ahrq_output = re.sub("\s+", " ", ahrq_output)
                    ahrq_output = ahrq_output.replace("```json ", "")
                    ahrq_output = ahrq_output.replace("```", "")

                    m3 = [
                        {"role": "system", "content": doctor_agent.discharge_system_prompt(ahrq_output)},
                        {"role": "user", "content": f"\nHere is a history of your dialogue: " + doctor_agent.agent_hist + f"\n. Based on your dialogue, give a discharge summary of your conversation for the patient following this exmaple: {example}. Personalize it to the patient specificially and give the json in a single line:\n"}]
                    
                    if doctor_llm in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                        summary_out = openai.chat.completions.create(
                            model=str_to_model[doctor_llm],
                            messages=m3,
                            response_format={ "type": "json_object" }
                        ) 
                    else:
                        summary_out = openai.chat.completions.create(
                            model=str_to_model[doctor_llm],
                            messages=m3,
                            response_format={ "type": "json_object" }
                        ) 
                    
                    summary_out = summary_out.choices[0].message.content
                elif doctor_llm == "mistral":
                    client = openai.OpenAI(
                        base_url=f"http://localhost:{VLLM_PORT}/v1",
                        api_key="-",
                    )
                    m2 = [
                        {"role": "user", "content": doctor_agent.system_prompt() + "\n" + f"\nHere is a history of your dialogue: " + doctor_agent.agent_hist + f"\n Generate an after visit plan based on your interactions according to the following example plan in a json format for each subheading given in the following example: {ahrqexample} .\n\n Be sure to list all mediciations."}]
                    ahrq_schema = AHRQ.model_json_schema()
                    ahrq_output = client.chat.completions.create(
                        model=LOCAL_MODEL,
                        messages=m2,
                        extra_body={"guided_json": ahrq_schema},
                    )
                    ahrq_output = ahrq_output.choices[0].message.content
                    ahrq_output = re.sub("\s+", " ", ahrq_output)
                    ahrq_output = ahrq_output.replace("```json ", "")
                    ahrq_output = ahrq_output.replace("```", "")

                    m3 = [
                        {"role": "user", "content":  doctor_agent.system_prompt() + "\n" + f"\b Here is the AHRQ plan you made: {ahrq_output} \nHere is a history of your dialogue: " + doctor_agent.agent_hist + f"\n. Based on your dialogue, give a discharge summary of your conversation for the patient following this example: {example}. Personalize it to the patient specificially and give the JSON in a single line.\n"}]
                    discharge_schema = DischargeInfo.model_json_schema()
                    summary_out = client.chat.completions.create(
                        model=LOCAL_MODEL,
                        messages=m3,
                        extra_body={"guided_json": discharge_schema},
                    )
                    summary_out = summary_out.choices[0].message.content

                else:
                    client = openai.OpenAI(
                        base_url=f"http://localhost:{VLLM_PORT}/v1",
                        api_key="-",
                    )
                    m2 = [
                        {"role": "system", "content": doctor_agent.system_prompt()},
                        {"role": "user", "content": f"\nHere is a history of your dialogue: " + doctor_agent.agent_hist + f"\n Generate an after visit plan based on your interactions according to the following example plan in a json format for each subheading given in the following example: {ahrqexample} .\n\n Be sure to list all mediciations."}]
                    ahrq_schema = AHRQ.model_json_schema()

                    ahrq_output = ""
                    if doctor_llm == "qwen3":
                        ahrq_output = client.chat.completions.create(
                            model=LOCAL_MODEL,
                            messages=m2,
                            extra_body={"guided_json": ahrq_schema, "chat_template_kwargs": {"enable_thinking": False}},
                        )
                    else:
                        ahrq_output = client.chat.completions.create(
                            model=LOCAL_MODEL,
                            messages=m2,
                            extra_body={"guided_json": ahrq_schema},
                        )
                    ahrq_output = ahrq_output.choices[0].message.content
                    ahrq_output = re.sub("\s+", " ", ahrq_output)
                    ahrq_output = ahrq_output.replace("```json ", "")
                    ahrq_output = ahrq_output.replace("```", "")

                    m3 = [
                        {"role": "system", "content": doctor_agent.system_prompt()},
                        {"role": "user", "content": f"\b Here is the AHRQ plan you made: {ahrq_output} \nHere is a history of your dialogue: " + doctor_agent.agent_hist + f"\n. Based on your dialogue, give a discharge summary of your conversation for the patient following this example: {example}. Personalize it to the patient specificially and give the JSON in a single line.\n"}]
                    discharge_schema = DischargeInfo.model_json_schema()

                    summary_out = ""
                    if doctor_llm == "qwen3":
                        summary_out = client.chat.completions.create(
                            model=LOCAL_MODEL,
                            messages=m3,
                            extra_body={"guided_json": discharge_schema, "chat_template_kwargs": {"enable_thinking": False}},
                        )
                    else:

                        summary_out = client.chat.completions.create(
                            model=LOCAL_MODEL,
                            messages=m3,
                            extra_body={"guided_json": discharge_schema},
                        )
                    summary_out = summary_out.choices[0].message.content

                
                with open(f"{output_file}/plan.jsonl", "a+") as f:
                    f.write(ahrq_output + "\n")

                with open(f"{output_file}/summary.jsonl", "a+") as f:
                    f.write(summary_out + "\n")


                with open(f"{output_file}/history.csv", 'a+', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    log_text = str(doctor_agent.log)
                    writer.writerow([log_text])

                break
            else:
                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                print("Patient [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
            # Prevent API timeouts
            #time.sleep(1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key')
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    # parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None')
    parser.add_argument('--doctor_llm', type=str, default='gpt4')
    parser.add_argument('--patient_llm', type=str, default='gpt4')
    parser.add_argument('--agent_dataset', type=str, default='MedQA') # MedQA, MIMICIV
    parser.add_argument('--num_scenarios', type=int, default=15, required=False, help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=50, required=False, help='Number of inferences between patient and doctor')

    parser.add_argument('--output_file', type=str, default=None, required=False)
    parser.add_argument('--model_file', type=str, default=None, required=False)
    parser.add_argument('--vllm_port', type=str, default='8000', required=False)

    args = parser.parse_args()

    LOCAL_MODEL = args.model_file
    VLLM_PORT   = args.vllm_port

    import os
    output_dir = f"{args.output_file}"
    os.makedirs(output_dir, exist_ok=True)

    main(args.openai_api_key, args.inf_type, args.doctor_bias, args.patient_bias, args.doctor_llm, args.patient_llm, args.num_scenarios, args.agent_dataset, args.total_inferences,  args.output_file)
