import { useEffect, useMemo, useRef, useState } from "react";
import "@/App.css";
import { CVPage } from "@/pages/CVPage";

type AppCommand = "start navigation" | "stop navigation" | "repeat" | "help";
type Page = "home" | "camera";

type LatLon = {
  latitude: number;
  longitude: number;
};

type Destination = LatLon & {
  name: string;
};

type RouteStep = {
  instruction: string;
  distanceMeters: number;
  location: LatLon;
};

type NominatimResult = {
  display_name: string;
  lat: string;
  lon: string;
};

type PhotonResult = {
  features?: Array<{
    properties?: {
      name?: string;
      city?: string;
      state?: string;
      country?: string;
    };
    geometry?: {
      coordinates?: [number, number];
    };
  }>;
};

type OsrmResponse = {
  code: string;
  routes?: Array<{
    legs: Array<{
      steps: Array<{
        distance: number;
        name: string;
        maneuver: {
          type: string;
          modifier?: string;
          bearing_after?: number;
          location: [number, number];
        };
      }>;
    }>;
  }>;
};

type SpeechRecognitionAlternative = {
  transcript: string;
};

type SpeechRecognitionResultLike = {
  isFinal: boolean;
  0?: SpeechRecognitionAlternative;
};

type SpeechRecognitionEventLike = Event & {
  results: ArrayLike<SpeechRecognitionResultLike>;
};

interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: ((event: Event) => void) | null;
  onend: ((event: Event) => void) | null;
  onerror: ((event: Event & { error?: string }) => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  start: () => void;
  stop: () => void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

const COMMANDS: AppCommand[] = [
  "start navigation",
  "stop navigation",
  "repeat",
  "help",
];

const DESTINATION_COMMAND_PREFIXES = [
  "set destination to",
  "navigate to",
  "go to",
  "take me to",
];

function getSpeechRecognitionConstructor() {
  return window.SpeechRecognition ?? window.webkitSpeechRecognition ?? null;
}

function App() {
  const [activePage, setActivePage] = useState<Page>("home");
  const [isListening, setIsListening] = useState(false);
  const [lastTranscript, setLastTranscript] = useState("-");
  const [navigationEnabled, setNavigationEnabled] = useState(false);
  const [lastGuidance, setLastGuidance] = useState(
    "Say 'help' to hear available commands.",
  );
  const [destinationQuery, setDestinationQuery] = useState("");
  const [destination, setDestination] = useState<Destination | null>(null);
  const [currentLocation, setCurrentLocation] = useState<LatLon | null>(null);
  const [currentLocationAccuracyMeters, setCurrentLocationAccuracyMeters] =
    useState<number | null>(null);
  const [routeSteps, setRouteSteps] = useState<RouteStep[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isRouting, setIsRouting] = useState(false);
  const [locationPermissionGranted, setLocationPermissionGranted] =
    useState(false);

  const activeNavigationWatchRef = useRef<number | null>(null);
  const passiveLocationWatchRef = useRef<number | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const lastStepSpokenRef = useRef<number>(-1);
  const routeStepsRef = useRef<RouteStep[]>([]);
  const currentStepIndexRef = useRef(0);
  const navigationEnabledRef = useRef(false);
  const currentLocationRef = useRef<LatLon | null>(null);
  const currentLocationAccuracyRef = useRef<number | null>(null);
  const destinationRef = useRef<Destination | null>(null);
  const destinationQueryRef = useRef("");
  const lastGuidanceRef = useRef(lastGuidance);

  const spokenState = useMemo(() => {
    return navigationEnabled ? "Navigation is ON." : "Navigation is OFF.";
  }, [navigationEnabled]);

  const speak = (text: string) => {
    if (!("speechSynthesis" in window)) {
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  const formatDistance = (distanceMeters: number): string => {
    if (distanceMeters < 10) {
      return "now";
    }
    if (distanceMeters < 1000) {
      return `${Math.round(distanceMeters)} meters`;
    }
    return `${(distanceMeters / 1000).toFixed(1)} kilometers`;
  };

  const bearingToDirection = (bearing: number): string => {
    const normalized = ((bearing % 360) + 360) % 360;
    const directions = [
      "north",
      "northeast",
      "east",
      "southeast",
      "south",
      "southwest",
      "west",
      "northwest",
    ];

    const index = Math.round(normalized / 45) % 8;
    return directions[index];
  };

  const getInstruction = (
    type: string,
    modifier: string | undefined,
    roadName: string,
    distanceMeters: number,
    bearingAfter?: number,
  ): string => {
    const roadPart = roadName ? ` onto ${roadName}` : "";
    const modifierText = modifier ? ` ${modifier}` : "";
    const distancePart = formatDistance(distanceMeters);

    if (type === "arrive") {
      return "You have arrived.";
    }
    if (type === "depart") {
      const heading =
        typeof bearingAfter === "number"
          ? bearingToDirection(bearingAfter)
          : modifier ?? "straight";
      return `Start by heading ${heading}${roadPart}.`;
    }
    if (type === "turn") {
      return `In ${distancePart}, turn${modifierText}${roadPart}.`;
    }
    if (type === "new name" || type === "continue") {
      return `Continue ${modifier ?? "straight"}${roadPart}.`;
    }
    if (type === "roundabout") {
      return `In ${distancePart}, take the roundabout${roadPart}.`;
    }
    return `In ${distancePart}, keep moving${roadPart}.`;
  };

  const distanceMetersBetween = (from: LatLon, to: LatLon): number => {
    const earthRadiusMeters = 6371000;
    const lat1 = (from.latitude * Math.PI) / 180;
    const lat2 = (to.latitude * Math.PI) / 180;
    const latDelta = ((to.latitude - from.latitude) * Math.PI) / 180;
    const lonDelta = ((to.longitude - from.longitude) * Math.PI) / 180;

    const a =
      Math.sin(latDelta / 2) * Math.sin(latDelta / 2) +
      Math.cos(lat1) * Math.cos(lat2) * Math.sin(lonDelta / 2) * Math.sin(lonDelta / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return earthRadiusMeters * c;
  };

  const setLocation = (coords: GeolocationCoordinates) => {
    const nextLocation = {
      latitude: coords.latitude,
      longitude: coords.longitude,
    };
    const nextAccuracy =
      typeof coords.accuracy === "number" ? coords.accuracy : null;

    currentLocationRef.current = nextLocation;
    currentLocationAccuracyRef.current = nextAccuracy;
    setCurrentLocation(nextLocation);
    setCurrentLocationAccuracyMeters(nextAccuracy);
    setLocationPermissionGranted(true);
  };

  const getCurrentPosition = (options: PositionOptions) => {
    return new Promise<GeolocationPosition>((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, options);
    });
  };

  const getLocationErrorMessage = (error: GeolocationPositionError) => {
    if (error.code === 1) {
      return "Location permission is required. Please allow browser location access.";
    }

    if (error.code === 2) {
      return "Location is currently unavailable. Move to an open area or check GPS settings.";
    }

    return "Timed out while waiting for GPS fix. Try again in a moment.";
  };

  const getCurrentLocationWithPermission = async (): Promise<LatLon | null> => {
    if (!("geolocation" in navigator)) {
      setLastGuidance("Geolocation is not available in this browser.");
      speak("Geolocation is not available in this browser.");
      return null;
    }

    const cachedLocation = currentLocationRef.current;
    const cachedAccuracy = currentLocationAccuracyRef.current;
    if (cachedLocation && (cachedAccuracy === null || cachedAccuracy <= 120)) {
      return cachedLocation;
    }

    try {
      const highAccuracyPosition = await getCurrentPosition({
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 10000,
      });

      setLocation(highAccuracyPosition.coords);
      return {
        latitude: highAccuracyPosition.coords.latitude,
        longitude: highAccuracyPosition.coords.longitude,
      };
    } catch (highAccuracyError) {
      const error = highAccuracyError as GeolocationPositionError;

      if (error.code === 1) {
        setLocationPermissionGranted(false);
      }

      if (error.code === 3 && currentLocationRef.current) {
        setLastGuidance("Using last known location while GPS refines.");
        return currentLocationRef.current;
      }

      try {
        const fallbackPosition = await getCurrentPosition({
          enableHighAccuracy: false,
          timeout: 10000,
          maximumAge: 60000,
        });

        setLocation(fallbackPosition.coords);
        return {
          latitude: fallbackPosition.coords.latitude,
          longitude: fallbackPosition.coords.longitude,
        };
      } catch (fallbackError) {
        const finalError = fallbackError as GeolocationPositionError;
        if (finalError.code === 1) {
          setLocationPermissionGranted(false);
        }

        const message = getLocationErrorMessage(finalError);
        setLastGuidance(message);
        speak(message);
        return null;
      }
    }
  };

  const searchDestination = async (inputQuery?: string): Promise<Destination | null> => {
    const query = (inputQuery ?? destinationQuery).trim();
    if (!query) {
      setLastGuidance("Type a destination first.");
      return null;
    }

    setDestinationQuery(query);

    try {
      setIsRouting(true);
      let newDestination: Destination | null = null;

      const nominatimUrl =
        "https://nominatim.openstreetmap.org/search?format=jsonv2&addressdetails=1&limit=1" +
        `&q=${encodeURIComponent(query)}`;

      const nominatimResponse = await fetch(nominatimUrl, {
        headers: {
          Accept: "application/json",
        },
      });

      if (nominatimResponse.ok) {
        const nominatimData = (await nominatimResponse.json()) as NominatimResult[];
        if (nominatimData.length) {
          const first = nominatimData[0];
          newDestination = {
            name: first.display_name,
            latitude: Number(first.lat),
            longitude: Number(first.lon),
          };
        }
      }

      if (!newDestination) {
        const photonUrl = `https://photon.komoot.io/api/?limit=1&q=${encodeURIComponent(query)}`;
        const photonResponse = await fetch(photonUrl, {
          headers: { Accept: "application/json" },
        });

        if (photonResponse.ok) {
          const photonData = (await photonResponse.json()) as PhotonResult;
          const feature = photonData.features?.[0];
          const coordinates = feature?.geometry?.coordinates;

          if (coordinates) {
            const displayName = [
              feature.properties?.name,
              feature.properties?.city,
              feature.properties?.state,
              feature.properties?.country,
            ]
              .filter(Boolean)
              .join(", ");

            newDestination = {
              name: displayName || query,
              latitude: coordinates[1],
              longitude: coordinates[0],
            };
          }
        }
      }

      if (!newDestination) {
        setDestination(null);
        setLastGuidance("Destination not found. Try a more specific place name.");
        speak("Destination not found. Try a more specific place name.");
        return null;
      }

      setDestination(newDestination);
      setLastGuidance(`Destination set: ${newDestination.name}`);
      speak("Destination found.");
      return newDestination;
    } catch {
      setLastGuidance("Failed to search destination.");
      speak("Failed to search destination.");
      return null;
    } finally {
      setIsRouting(false);
    }
  };

  const buildRoute = async (start: LatLon, end: Destination): Promise<RouteStep[] | null> => {
    const url =
      "https://router.project-osrm.org/route/v1/foot/" +
      `${start.longitude},${start.latitude};${end.longitude},${end.latitude}` +
      "?overview=false&alternatives=false&steps=true";

    const response = await fetch(url, { headers: { Accept: "application/json" } });
    const data = (await response.json()) as OsrmResponse;

    if (data.code !== "Ok" || !data.routes?.length || !data.routes[0].legs.length) {
      return null;
    }

    const stepData = data.routes[0].legs[0].steps;
    return stepData.map((step) => {
      const maneuverLocation: LatLon = {
        latitude: step.maneuver.location[1],
        longitude: step.maneuver.location[0],
      };

      return {
        instruction: getInstruction(
          step.maneuver.type,
          step.maneuver.modifier,
          step.name,
          step.distance,
          step.maneuver.bearing_after,
        ),
        distanceMeters: step.distance,
        location: maneuverLocation,
      };
    });
  };

  const stopMacroNavigation = () => {
    if (activeNavigationWatchRef.current !== null) {
      navigator.geolocation.clearWatch(activeNavigationWatchRef.current);
      activeNavigationWatchRef.current = null;
    }

    setNavigationEnabled(false);
    setLastGuidance("Navigation stopped.");
  };

  const handleLocationUpdate = (nextLocation: LatLon) => {
    currentLocationRef.current = nextLocation;
    setCurrentLocation(nextLocation);

    if (!navigationEnabledRef.current || currentStepIndexRef.current >= routeStepsRef.current.length) {
      return;
    }

    const nextStep = routeStepsRef.current[currentStepIndexRef.current];
    const distanceToStep = distanceMetersBetween(nextLocation, nextStep.location);

    if (distanceToStep <= 12) {
      const upcomingIndex = currentStepIndexRef.current + 1;

      if (upcomingIndex >= routeStepsRef.current.length) {
        setCurrentStepIndex(upcomingIndex);
        setLastGuidance("You have arrived.");
        speak("You have arrived.");
        stopMacroNavigation();
        return;
      }

      setCurrentStepIndex(upcomingIndex);
    }
  };

  const startMacroNavigation = async () => {
    let selectedDestination = destinationRef.current;
    const pendingQuery = destinationQueryRef.current.trim();
    if (!selectedDestination && pendingQuery) {
      selectedDestination = await searchDestination(pendingQuery);
    }

    if (!selectedDestination) {
      setLastGuidance("Search and select a destination first.");
      speak("Search and select a destination first.");
      return;
    }

    try {
      setIsRouting(true);
      const start = await getCurrentLocationWithPermission();
      if (!start) {
        return;
      }

      const steps = await buildRoute(start, selectedDestination);
      if (!steps?.length) {
        setLastGuidance("Could not build walking route.");
        speak("Could not build walking route.");
        return;
      }

      setRouteSteps(steps);
      setCurrentStepIndex(0);
      lastStepSpokenRef.current = -1;
      setNavigationEnabled(true);
      setLastGuidance(steps[0].instruction);
      speak(steps[0].instruction);

      if (activeNavigationWatchRef.current !== null) {
        navigator.geolocation.clearWatch(activeNavigationWatchRef.current);
      }

      activeNavigationWatchRef.current = navigator.geolocation.watchPosition(
        (position) => {
          const nextLocation = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          };
          setCurrentLocationAccuracyMeters(
            typeof position.coords.accuracy === "number"
              ? position.coords.accuracy
              : null,
          );
          handleLocationUpdate(nextLocation);
        },
        (error) => {
          if (error.code === 1) {
            setLocationPermissionGranted(false);
          }

          setLastGuidance(getLocationErrorMessage(error));
        },
        {
          enableHighAccuracy: true,
          maximumAge: 3000,
          timeout: 15000,
        },
      );
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unknown error";
      setLastGuidance(`Failed to start navigation: ${reason}`);
      speak("Failed to start navigation. Check location and destination.");
    } finally {
      setIsRouting(false);
    }
  };

  const detectCommand = (transcript: string): AppCommand | null => {
    const normalized = transcript.toLowerCase().trim();
    for (const command of COMMANDS) {
      if (normalized.includes(command)) {
        return command;
      }
    }
    return null;
  };

  const detectDestinationCommand = (transcript: string): string | null => {
    const normalized = transcript.toLowerCase().trim();

    for (const prefix of DESTINATION_COMMAND_PREFIXES) {
      const index = normalized.indexOf(prefix);
      if (index !== -1) {
        const afterPrefix = transcript
          .slice(index + prefix.length)
          .replace(/^[\s,:-]+/, "")
          .replace(/[?.!,]+$/g, "")
          .trim();

        return afterPrefix.length ? afterPrefix : null;
      }
    }

    return null;
  };

  const applyCommand = (command: AppCommand) => {
    if (command === "start navigation") {
      void startMacroNavigation();
      return;
    }

    if (command === "stop navigation") {
      stopMacroNavigation();
      speak("Navigation stopped.");
      return;
    }

    if (command === "repeat") {
      speak(lastGuidanceRef.current);
      return;
    }

    setLastGuidance("Available commands: start navigation, stop navigation, repeat, help.");
    speak("Available commands: start navigation, stop navigation, repeat, help.");
  };

  const setupRecognition = () => {
    const Recognition = getSpeechRecognitionConstructor();
    if (!Recognition) {
      setLastGuidance("Speech recognition is not available in this browser.");
      speak("Speech recognition is not available on this device.");
      return null;
    }

    const recognition = new Recognition();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setLastGuidance("Listening. Say a command.");
      speak("Listening.");
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.onerror = (event) => {
      setIsListening(false);
      setLastGuidance(`Speech error: ${event.error ?? "unknown"}`);
    };

    recognition.onresult = (event) => {
      if (!event.results.length) {
        return;
      }

      const latestResult = event.results[event.results.length - 1];
      const transcript = latestResult?.[0]?.transcript?.trim();
      if (!transcript) {
        return;
      }

      setLastTranscript(transcript);
      if (!latestResult.isFinal) {
        return;
      }

      const destinationFromVoice = detectDestinationCommand(transcript);
      if (destinationFromVoice) {
        setLastGuidance(`Setting destination to ${destinationFromVoice}`);
        speak(`Setting destination to ${destinationFromVoice}`);
        void searchDestination(destinationFromVoice);
        return;
      }

      const command = detectCommand(transcript);
      if (command) {
        applyCommand(command);
        return;
      }

      setLastGuidance("Command not recognized. Say help.");
      speak("Command not recognized. Say help.");
    };

    return recognition;
  };

  const toggleListening = async () => {
    if (isListening) {
      recognitionRef.current?.stop();
      return;
    }

    if (!recognitionRef.current) {
      recognitionRef.current = setupRecognition();
    }

    if (!recognitionRef.current) {
      return;
    }

    try {
      recognitionRef.current.start();
    } catch {
      setLastGuidance("Unable to start listening. Try again.");
    }
  };

  useEffect(() => {
    if (!navigationEnabled || currentStepIndex >= routeSteps.length) {
      return;
    }

    if (lastStepSpokenRef.current === currentStepIndex) {
      return;
    }

    const step = routeSteps[currentStepIndex];
    setLastGuidance(step.instruction);
    speak(step.instruction);
    lastStepSpokenRef.current = currentStepIndex;
  }, [currentStepIndex, navigationEnabled, routeSteps]);

  useEffect(() => {
    routeStepsRef.current = routeSteps;
    currentStepIndexRef.current = currentStepIndex;
    navigationEnabledRef.current = navigationEnabled;
  }, [routeSteps, currentStepIndex, navigationEnabled]);

  useEffect(() => {
    destinationRef.current = destination;
    destinationQueryRef.current = destinationQuery;
    lastGuidanceRef.current = lastGuidance;
  }, [destination, destinationQuery, lastGuidance]);

  useEffect(() => {
    if (!("geolocation" in navigator)) {
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation(position.coords);
      },
      (error) => {
        if (error.code === 1) {
          setLocationPermissionGranted(false);
        }
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 10000,
      },
    );

    passiveLocationWatchRef.current = navigator.geolocation.watchPosition(
      (position) => {
        setLocation(position.coords);
      },
      (error) => {
        if (error.code === 1) {
          setLocationPermissionGranted(false);
        }
      },
      {
        enableHighAccuracy: true,
        maximumAge: 2000,
        timeout: 15000,
      },
    );
  }, []);

  useEffect(() => {
    return () => {
      if (activeNavigationWatchRef.current !== null) {
        navigator.geolocation.clearWatch(activeNavigationWatchRef.current);
      }
      if (passiveLocationWatchRef.current !== null) {
        navigator.geolocation.clearWatch(passiveLocationWatchRef.current);
      }
      recognitionRef.current?.stop();
      window.speechSynthesis.cancel();
    };
  }, []);

  return (
    <main className="app-page">
      <header className="app-header">
        <h1>Voice Navigation Control</h1>
        <p>{spokenState}</p>
      </header>

      <nav className="tab-row" aria-label="Main pages">
        <button
          type="button"
          className={activePage === "home" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActivePage("home")}
        >
          Home
        </button>
        <button
          type="button"
          className={activePage === "camera" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActivePage("camera")}
        >
          Camera + YOLO26
        </button>
      </nav>

      {activePage === "camera" ? (
        <CVPage />
      ) : (
        <section className="home-panel">
          <label htmlFor="destination" className="field-label">
            Destination:
          </label>
          <input
            id="destination"
            value={destinationQuery}
            onChange={(event) => setDestinationQuery(event.target.value)}
            placeholder="Type destination (e.g. Times Square)"
            className="destination-input"
          />

          <div className="action-row">
            <button
              type="button"
              onClick={() => {
                void searchDestination();
              }}
              disabled={isRouting}
            >
              Search
            </button>
            <button
              type="button"
              onClick={() => {
                void startMacroNavigation();
              }}
              disabled={isRouting}
              className="primary-btn"
            >
              Start Route
            </button>
          </div>

          <p>{destination ? `Destination: ${destination.name}` : "Destination: not set"}</p>
          <p>Last heard: {lastTranscript}</p>
          <p>Guidance: {lastGuidance}</p>
          <p>
            Step: {routeSteps.length ? `${Math.min(currentStepIndex + 1, routeSteps.length)}/${routeSteps.length}` : "-"}
          </p>
          <p>
            Location: {currentLocation ? `${currentLocation.latitude.toFixed(5)}, ${currentLocation.longitude.toFixed(5)}` : locationPermissionGranted ? "Waiting for GPS fix" : "Unknown"}
          </p>
          <p>
            GPS accuracy: {currentLocationAccuracyMeters !== null ? `${Math.round(currentLocationAccuracyMeters)} m` : "Unknown"}
          </p>

          <button
            type="button"
            onClick={() => {
              void toggleListening();
            }}
            className={isListening ? "voice-btn stop" : "voice-btn start"}
          >
            {isListening ? "Stop Listening" : "Start Listening"}
          </button>

          <p className="hint-text">
            Commands: start navigation, stop navigation, repeat, help, set destination to ...
          </p>
        </section>
      )}
    </main>
  );
}

export default App;
