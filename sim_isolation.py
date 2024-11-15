import pythia8
import pandas as pd
import numpy as np

# Initialize Pythia
pythia = pythia8.Pythia()

# Enable SoftQCD non-diffractive processes
pythia.readString("SoftQCD:nonDiffractive = on")

pythia.readString("9900022:new = gamma_dark DarkPhoton")  # Define new particle
pythia.readString("9900022:spinType = 1")                 # Spin-1 particle (vector boson)
pythia.readString("9900022:chargeType = 0")               # Neutral particle
pythia.readString("9900022:colType = 0")                  # Not colored
pythia.readString("9900022:m0 = 0.002")                   # Set mass to 2 MeV (above 1 MeV)
pythia.readString("9900022:tau0 = 0")                     # Zero lifetime, decays immediately
pythia.readString("9900022:mayDecay = on")                # Allow dark photon to decay
pythia.readString("9900022:isResonance = false")          # Not treated as a resonance
pythia.readString("9900022:addChannel = 1 1.0 0 11 -11")  # Dark photon -> e+ e-

pythia.readString("111:addChannel = 1 0.000001 0 22 9900022")  # pi0 -> gamma gamma_dark

pythia.init()

events_num = 10000
events = []

isolation_cone_radius = 0.29388785105159554
isolation_threshold = 0.20776536494954143

for i_event in range(events_num):
    if not pythia.next():
        continue

    particles = []

    dark_photon_produced = False

    for i in range(pythia.event.size()):
        particle = pythia.event[i]

        if particle.isFinal():
            # Collect particle properties
            particle_data = [
                particle.px(), particle.py(), particle.pz(), particle.e(),
                particle.pT(), particle.eta(), particle.phi(), particle.id()
            ]
            particles.append(particle_data)

        # Detect if Dark Photon was produced
        if particle.id() == 9900022:
            dark_photon_produced = True

    if len(particles) == 0:
        continue  # Skip event if no particles collected

    particles = np.array(particles)

    # Isolation calculations
    def calculate_isolation(target_particles, all_particles, cone_radius, isolation_threshold):
        if target_particles.shape[0] == 0:
            return 0, 0.0

        # Compute Delta Eta and Delta Phi differences using broadcasting
        delta_eta = target_particles[:, np.newaxis, 5] - all_particles[np.newaxis, :, 5]
        delta_phi = np.arctan2(
            np.sin(target_particles[:, np.newaxis, 6] - all_particles[np.newaxis, :, 6]),
            np.cos(target_particles[:, np.newaxis, 6] - all_particles[np.newaxis, :, 6])
        )

        # Compute Delta R
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

        # Mask to exclude particles within the cone but exclude self-comparison
        isolation_mask = (delta_r < cone_radius) & (delta_r > 0)

        # Calculate sum of pT of particles within the cone for each target particle
        isolation_sum = np.sum(all_particles[:, 4] * isolation_mask, axis=1)

        # Calculate isolation ratio
        isolation_ratio = isolation_sum / target_particles[:, 4]

        # Count isolated particles
        isolated_count = np.sum(isolation_ratio < isolation_threshold)

        # Calculate average isolation ratio
        avg_isolation_ratio = np.mean(isolation_ratio) if target_particles.shape[0] > 0 else 0.0

        return isolated_count, avg_isolation_ratio

    # Extract leptons and photons from particles array
    leptons_array = particles[np.abs(particles[:, 7]) == 11]  # e+ and e-
    photons_array = particles[particles[:, 7] == 22]

    # Calculate isolation for leptons and photons
    lepton_isolated_count, lepton_avg_isolation = calculate_isolation(
        leptons_array, particles, isolation_cone_radius, isolation_threshold
    )
    photon_isolated_count, photon_avg_isolation = calculate_isolation(
        photons_array, particles, isolation_cone_radius, isolation_threshold
    )

    # Store event data
    event_data = {
        "Event Number": i_event,
        "Lepton Isolated Count": lepton_isolated_count,
        "Lepton Avg Isolation": lepton_avg_isolation,
        "Photon Isolated Count": photon_isolated_count,
        "Photon Avg Isolation": photon_avg_isolation,
        "Dark Photon Produced": dark_photon_produced,
    }
    events.append(event_data)


