import optuna
import pythia8
import pandas as pd
import numpy as np

# Function to calculate isolation
def calculate_isolation(target_particles, all_particles, cone_radius, isolation_threshold):
    if target_particles.shape[0] == 0:
        return 0, 0.0

    delta_eta = target_particles[:, np.newaxis, 5] - all_particles[np.newaxis, :, 5]
    delta_phi = np.arctan2(
        np.sin(target_particles[:, np.newaxis, 6] - all_particles[np.newaxis, :, 6]),
        np.cos(target_particles[:, np.newaxis, 6] - all_particles[np.newaxis, :, 6])
    )

    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
    isolation_mask = (delta_r < cone_radius) & (delta_r > 0)

    isolation_sum = np.sum(all_particles[:, 4] * isolation_mask, axis=1)
    isolation_ratio = isolation_sum / target_particles[:, 4]

    isolated_count = np.sum(isolation_ratio < isolation_threshold)
    avg_isolation_ratio = np.mean(isolation_ratio) if target_particles.shape[0] > 0 else 0.0

    return isolated_count, avg_isolation_ratio

# Function to run Pythia simulation and calculate isolation metrics
def run_simulation(cone_radius, isolation_threshold, events_num=7000):
    pythia = pythia8.Pythia()
    pythia.readString("SoftQCD:nonDiffractive = on")
    pythia.readString("9900022:new = gamma_dark DarkPhoton")
    pythia.readString("9900022:spinType = 1")
    pythia.readString("9900022:chargeType = 0")
    pythia.readString("9900022:colType = 0")
    pythia.readString("9900022:m0 = 0.002")
    pythia.readString("9900022:tau0 = 0")
    pythia.readString("9900022:mayDecay = on")
    pythia.readString("9900022:addChannel = 1 1.0 0 11 -11")
    pythia.readString("111:addChannel = 1 0.000001 0 22 9900022")
    pythia.init()

    events = []
    for i_event in range(events_num):
        if not pythia.next():
            continue

        particles = []
        dark_photon_produced = False

        for i in range(pythia.event.size()):
            particle = pythia.event[i]
            if particle.isFinal():
                particle_data = [
                    particle.px(), particle.py(), particle.pz(), particle.e(),
                    particle.pT(), particle.eta(), particle.phi(), particle.id()
                ]
                particles.append(particle_data)
            if particle.id() == 9900022:
                dark_photon_produced = True

        if len(particles) == 0:
            continue

        particles = np.array(particles)

        leptons_array = particles[np.abs(particles[:, 7]) == 11]
        photons_array = particles[particles[:, 7] == 22]

        lepton_isolated_count, lepton_avg_isolation = calculate_isolation(
            leptons_array, particles, cone_radius, isolation_threshold
        )
        photon_isolated_count, photon_avg_isolation = calculate_isolation(
            photons_array, particles, cone_radius, isolation_threshold
        )

        event_data = {
            "Lepton Isolated Count": lepton_isolated_count,
            "Lepton Avg Isolation": lepton_avg_isolation,
            "Photon Isolated Count": photon_isolated_count,
            "Photon Avg Isolation": photon_avg_isolation,
            "Dark Photon Produced": dark_photon_produced,
        }
        events.append(event_data)

    df_events = pd.DataFrame(events)
    correlation_matrix = df_events.corr()
    correlations_with_target = correlation_matrix['Dark Photon Produced'].drop('Dark Photon Produced')
    sorted_correlations = correlations_with_target.abs().sort_values(ascending=False)
    
    if len(sorted_correlations) > 0:
        return sorted_correlations.iloc[0]
    else:
        return 0.0

# Optuna objective function to optimize the parameters
def objective(trial):
    cone_radius = trial.suggest_float("isolation_cone_radius", 0.1, 1.0)
    isolation_threshold = trial.suggest_float("isolation_threshold", 0.01, 0.5)
    
    correlation = run_simulation(cone_radius, isolation_threshold)
    return correlation

# Create Optuna study to maximize the correlation
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Display best parameters
best_params = study.best_params
best_value = study.best_value
print(f"Best Isolation Cone Radius: {best_params['isolation_cone_radius']}")
print(f"Best Isolation Threshold: {best_params['isolation_threshold']}")
print(f"Best Correlation Value: {best_value}")

# Best Isolation Cone Radius: 0.29388785105159554
# Best Isolation Threshold: 0.20776536494954143
# Best Correlation Value: 0.019546099133488305