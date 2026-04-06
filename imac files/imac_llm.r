/* imac_llm.r — Resource file for iMac G3 LLM
 *
 * Sets the SIZE resource so Mac OS 8.5 gives the app
 * enough memory to load the model weights (~1 MB).
 *
 * To use: uncomment the target_sources line in CMakeLists.txt
 *
 * Note: The SIZE resource may not survive MacBinary transfer
 * via FTP. If the app fails to allocate memory, manually set
 * the Preferred Size to 3000 KB in File -> Get Info -> Memory.
 */

#include "Retro68.r"

resource 'SIZE' (-1) {
    reserved,
    acceptSuspendResumeEvents,
    reserved,
    canBackground,
    doesActivateOnFGSwitch,
    backgroundAndForeground,
    dontGetFrontClicks,
    ignoreChildDiedEvents,
    is32BitCompatible,
    isHighLevelEventAware,
    onlyLocalHLEvents,
    notStationeryAware,
    dontUseTextEditServices,
    reserved,
    reserved,
    reserved,
    4194304,    /* preferred size: 4 MB */
    2097152     /* minimum size: 2 MB */
};
