// config_map_helpers.go
// CHECKED
package mypriorityoptimizer

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
)

// -------------------------
// listConfigMaps
// -------------------------

// listConfigMaps lists config maps in the namespace with the given label key,
// sorted by creation timestamp descending.
func listConfigMaps(nsLister corev1listers.ConfigMapNamespaceLister, labelKey string) ([]v1.ConfigMap, error) {
	// List with label selector
	sel := labels.SelectorFromSet(labels.Set{labelKey: "true"})
	items, err := nsLister.List(sel)
	if err != nil {
		return nil, err
	}

	// Make a copy and sort by creation timestamp descending
	cms := make([]v1.ConfigMap, len(items))
	for i := range items {
		cms[i] = *items[i].DeepCopy()
	}

	// Sort by creation timestamp descending
	sort.Slice(cms, func(i, j int) bool {
		return cms[i].CreationTimestamp.Time.After(cms[j].CreationTimestamp.Time)
	})
	return cms, nil
}

// -------------------------
// pruneConfigMaps
// -------------------------

// pruneConfigMaps keeps first K newest config maps with label, deletes the rest.
func pruneConfigMaps(
	ctx context.Context,
	cms corev1client.ConfigMapInterface,
	nsLister corev1listers.ConfigMapNamespaceLister,
	labelKey string,
	keep int,
) error {
	// Nothing to do
	if keep <= 0 {
		return nil
	}

	// List config maps with label
	items, err := listConfigMaps(nsLister, labelKey)
	if err != nil || len(items) <= keep {
		return err
	}

	// Delete older ones
	for i := keep; i < len(items); i++ {
		if err := cms.Delete(ctx, items[i].Name, metav1.DeleteOptions{}); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return err
		}
	}
	return nil
}

// -------------------------
// marshalJsonIndented
// -------------------------

// marshalJsonIndented marshals an object to JSON with indentation.
func marshalJsonIndented(v any) ([]byte, error) {
	return json.MarshalIndent(v, "", "  ")
}

// -------------------------
// jsonString
// -------------------------

// marshalToJsonString pretty-prints v to JSON and returns it as a string.
func marshalToJsonString(v any) (string, error) {
	b, err := marshalJsonIndented(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// -------------------------
// patchDataString
// -------------------------

// patchDataString patches a single DataKey with the given raw JSON string.
func (d ConfigMapDoc) patchDataString(
	ctx context.Context,
	cms corev1client.ConfigMapInterface,
	raw string,
) error {
	// Create merge patch
	patch := []byte(fmt.Sprintf(`{"data":{"%s":%q}}`, d.DataKey, raw))
	_, err := cms.Patch(
		ctx,
		d.Name,
		types.MergePatchType,
		patch,
		metav1.PatchOptions{},
	)
	return err
}

// -------------------------
// ensureJson
// -------------------------

// ensureJson creates or updates config map, storing data as JSON at DataKey.
func (d ConfigMapDoc) ensureJson(
	ctx context.Context,
	cms corev1client.ConfigMapInterface,
	data any,
) error {
	// Marshal to JSON bytes
	b, err := marshalJsonIndented(data)
	if err != nil {
		return err
	}

	// Get existing
	cm, err := cms.Get(ctx, d.Name, metav1.GetOptions{})
	switch {
	case apierrors.IsNotFound(err): // create new
		cm = &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      d.Name,
				Namespace: d.Namespace,
				Labels:    map[string]string{d.LabelKey: "true"},
			},
			Data: map[string]string{d.DataKey: string(b)},
		}
		_, err = cms.Create(ctx, cm, metav1.CreateOptions{})
		return err

	case err != nil:
		return err

	default: // update existing
		if cm.Data == nil {
			cm.Data = map[string]string{}
		}
		cm.Data[d.DataKey] = string(b)
		_, err = cms.Update(ctx, cm, metav1.UpdateOptions{})
		return err
	}
}

// -------------------------
// patchJson
// -------------------------

// patchJson patches only DataKey via merge patch.
func (d ConfigMapDoc) patchJson(
	ctx context.Context,
	cms corev1client.ConfigMapInterface,
	v any,
) error {
	// Marshal to JSON string
	jsonStr, err := marshalToJsonString(v)
	if err != nil {
		return err
	}
	// Patch data key
	return d.patchDataString(ctx, cms, jsonStr)
}

// -------------------------
// readJson
// -------------------------

// readJson reads DataKey as JSON bytes.
func (d ConfigMapDoc) readJson(
	nsLister corev1listers.ConfigMapNamespaceLister,
) (raw []byte, found bool, err error) {
	// get config map
	cm, err := nsLister.Get(d.Name)

	// NotFound => treat as missing (no error)
	if apierrors.IsNotFound(err) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	if cm == nil {
		return nil, false, nil
	}

	// get data key
	return []byte(cm.Data[d.DataKey]), true, nil
}

// -------------------------
// mutateJson
// -------------------------

// mutateJson loads -> mutates -> patches an array JSON.
func mutateJson[T any](
	ctx context.Context,
	cms corev1client.ConfigMapInterface,
	nsLister corev1listers.ConfigMapNamespaceLister,
	doc ConfigMapDoc,
	f func(existing []T) ([]T, error),
) error {
	// read existing
	raw, found, err := doc.readJson(nsLister)
	if err != nil || !found {
		return err // no-op on missing, propagate error
	}

	// unmarshal existing array
	var arr []T
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &arr)
	}

	// mutate
	out, err := f(arr)
	if err != nil || out == nil { // allow nil => “no change”
		return err
	}
	return doc.patchJson(ctx, cms, out)
}

// -------------------------
// mutateRaw
// -------------------------

// mutateRaw loads JSON string at DataKey, mutates it, and writes result back.
func (d ConfigMapDoc) mutateRaw(
	ctx context.Context,
	cms corev1client.ConfigMapInterface,
	nsLister corev1listers.ConfigMapNamespaceLister,
	mutate func(raw []byte) ([]byte, error),
) error {
	// Read existing
	raw, found, err := d.readJson(nsLister)
	if err != nil || !found {
		return err // missing => no-op
	}

	// Mutate
	newRaw, err := mutate(raw)
	if err != nil || newRaw == nil {
		return err // nil => no-op
	}

	// Patch back
	return d.patchDataString(ctx, cms, string(newRaw))
}
