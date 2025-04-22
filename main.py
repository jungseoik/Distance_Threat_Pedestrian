from models.threat import ThreatVideoDiscriminator

model = ThreatVideoDiscriminator(use_classifier=True, output_path="output/result1.mp4")
model.process_video("assets/threat_1.mp4")

model = ThreatVideoDiscriminator(use_classifier=False , output_path="output/result2.mp4")
model.process_video("assets/threat_1.mp4")
