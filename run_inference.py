import sys
sys.path.insert(0, "D:/Hackathon/VoiceAIHackathon2.0-Oracle/src")

from btc.evaluation.inference import generate_inference_report

report = generate_inference_report(
    data_dir="data",
    bundle_path="artifacts/model_bundle",
    output_path="artifacts/inference_report.json",
    n_sellers=100,
)

print(f"Tested {report['n_sellers_tested']} sellers")
print(f"Average predicted best hour: {report['summary']['avg_predicted_best_hour']:.1f}")
print(f"Average expected reward: {report['summary']['avg_expected_reward']:.3f}")
print(f"Average prior weight: {report['summary']['avg_prior_weight']:.3f}")
print(f"Sellers with secondary: {report['summary']['sellers_with_secondary']}")
print(f"Segments: {len(report['summary']['segment_distribution'])}")
